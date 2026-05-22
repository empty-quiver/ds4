#!/usr/bin/env python3
"""Remote A/B benchmark for DS4 MTP speculative decoding.

This script is intentionally defensive: every SSH, readiness, and generation
operation has a timeout; server restarts wait for the old ds4-server process
and port listener to drain before launching the next variant; and the final
server state is restored even if one side of the benchmark fails.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shlex
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass


DEFAULT_REMOTE_ENV = {
    "DS4_CPU_AFFINITY": "1",
    "DS4_CPU_AFFINITY_LIST": "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15",
    "DS4_CPU_MOE_ROW_CHUNK": "8",
    "DS4_CUDA_DIRECT_MODEL": "1",
    "DS4_CUDA_PARTIAL_WEIGHT_CACHE": "1",
    "DS4_CUDA_WEIGHT_CACHE_LIMIT_GB": "10",
    "DS4_CUDA_REQUIRE_DENSE_WEIGHT_CACHE": "1",
    "DS4_CUDA_Q8_F16_CACHE_RESERVE_MB": "4096",
    "DS4_CUDA_DYNAMIC_EXPERTS": "1",
    "DS4_CUDA_DYNAMIC_EXPERT_CACHE_GB": "4",
    "DS4_CUDA_DYNAMIC_EXPERT_POLICY": "lru",
    "DS4_CUDA_DYNAMIC_EXPERT_MAX_EVICTIONS": "64",
    "DS4_CUDA_DYNAMIC_EXPERT_GROUP_SIZE": "4",
    "DS4_CUDA_DYNAMIC_EXPERT_GROUP_EVICTION": "1",
    "DS4_CUDA_DYNAMIC_EXPERT_DECODE_EAGER": "1",
    "DS4_CUDA_DYNAMIC_EXPERT_DECODE_EVICT": "0",
    "DS4_CUDA_DYNAMIC_EXPERT_DECODE_MIN_SCORE": "8",
    "DS4_CUDA_DYNAMIC_EXPERT_DECODE_MAINTENANCE_INTERVAL": "1",
    "DS4_CUDA_DYNAMIC_EXPERT_DECODE_MAX_PROMOTIONS": "64",
    "DS4_CUDA_DYNAMIC_EXPERT_DECODE_GROUP_SIZE": "4",
    "DS4_CUDA_LAYERWISE_PREFILL_STAGING": "0",
}


@dataclass(frozen=True)
class Sample:
    tokens: int
    seconds: float
    tok_s: float
    finish_reason: str
    output_hash: str


@dataclass(frozen=True)
class Variant:
    label: str
    mode: str
    mtp_draft: int | None = None
    mtp_margin: float | None = None
    strict: bool = False
    mtp_gpu: bool = False
    mtp_cpu: bool = False


VARIANTS: dict[str, Variant] = {
    "nomtp": Variant(label="nomtp", mode="nomtp"),
    "mtp-fast-m3": Variant(label="mtp-fast-m3", mode="mtp", mtp_margin=3.0),
    "mtp-fast-m0": Variant(label="mtp-fast-m0", mode="mtp", mtp_margin=0.0),
    "mtp-strict": Variant(label="mtp-strict", mode="mtp", strict=True),
    "mtp-fast-m3-gpuw": Variant(label="mtp-fast-m3-gpuw", mode="mtp", mtp_margin=3.0, mtp_gpu=True),
    "mtp-strict-gpuw": Variant(label="mtp-strict-gpuw", mode="mtp", strict=True, mtp_gpu=True),
    "mtp-fast-m0-cpu": Variant(label="mtp-fast-m0-cpu", mode="mtp", mtp_margin=0.0, mtp_cpu=True),
    "mtp-fast-m3-cpu": Variant(label="mtp-fast-m3-cpu", mode="mtp", mtp_margin=3.0, mtp_cpu=True),
    "mtp-fast-m6-cpu": Variant(label="mtp-fast-m6-cpu", mode="mtp", mtp_margin=6.0, mtp_cpu=True),
    "mtp-strict-cpu": Variant(label="mtp-strict-cpu", mode="mtp", strict=True, mtp_cpu=True),
}

MTP_STATS_RE = re.compile(
    r"\b(attempts|strict|fast|no_draft|first_hit|first_miss|drafted|accepted|"
    r"full|partial|zero|margin_skip|micro|exact2|seq|seq_fallback|draft_fail|"
    r"verify_fail|prefix1|exact_replay)=(\d+)"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ssh-host", default="eve@eve-lambda-vector.local")
    p.add_argument("--base-url", default="http://10.0.1.124:8001/v1")
    p.add_argument("--remote-build-dir", default="/tmp/ds4-decode-tier-build")
    p.add_argument("--remote-model", default="/home/eve/ds4/ds4flash.gguf")
    p.add_argument(
        "--remote-mtp-model",
        default="/home/eve/ds4/gguf/DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf",
    )
    p.add_argument("--remote-log-dir", default="/tmp")
    p.add_argument("--port", type=int, default=8001)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--ctx", type=int, default=262144)
    p.add_argument("--threads", type=int, default=16)
    p.add_argument("--server-tokens", type=int, default=16384)
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--warmups", type=int, default=1)
    p.add_argument("--mtp-draft", type=int, default=2)
    p.add_argument("--ready-timeout", type=float, default=90.0)
    p.add_argument("--request-timeout", type=float, default=240.0)
    p.add_argument("--ssh-timeout", type=float, default=60.0)
    p.add_argument("--shutdown-timeout", type=float, default=20.0)
    p.add_argument("--tmux-session", default="ds4-bench")
    p.add_argument("--final-session", default="ds4-256k")
    p.add_argument(
        "--restore",
        choices=("mtp", "mtp-cpu", "nomtp", "none"),
        default="mtp",
        help="Server variant to leave running after the benchmark.",
    )
    p.add_argument(
        "--variants",
        default="nomtp,mtp-fast-m3,mtp-fast-m0,mtp-strict",
        help=(
            "Comma-separated variants to run. Choices: "
            + ",".join(sorted(VARIANTS))
        ),
    )
    p.add_argument(
        "--prompt",
        default=(
            "Write a compact numbered list of CUDA kernel optimization ideas "
            "for a sparse MoE attention runtime. Use short clauses only. Keep "
            "going until the token limit stops you."
        ),
    )
    return p.parse_args()


def selected_variants(args: argparse.Namespace) -> list[Variant]:
    labels = [item.strip() for item in args.variants.split(",") if item.strip()]
    if not labels:
        raise ValueError("--variants must name at least one variant")
    variants = []
    for label in labels:
        if label not in VARIANTS:
            raise ValueError(f"unknown variant {label!r}; choices: {', '.join(sorted(VARIANTS))}")
        base = VARIANTS[label]
        if base.mode == "mtp":
            variants.append(Variant(
                label=base.label,
                mode=base.mode,
                mtp_draft=args.mtp_draft if base.mtp_draft is None else base.mtp_draft,
                mtp_margin=base.mtp_margin,
                strict=base.strict,
                mtp_gpu=base.mtp_gpu,
                mtp_cpu=base.mtp_cpu,
            ))
        else:
            variants.append(base)
    return variants


def print_event(event: str, **values: object) -> None:
    row = {"event": event, **values}
    print(json.dumps(row, sort_keys=True), flush=True)


def ssh(args: argparse.Namespace, script: str, *, timeout: float | None = None,
        check: bool = True) -> subprocess.CompletedProcess[str]:
    cmd = [
        "ssh",
        "-o", "BatchMode=yes",
        "-o", "ConnectTimeout=8",
        args.ssh_host,
        "bash", "-s",
    ]
    proc = subprocess.run(
        cmd,
        input=script,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout if timeout is not None else args.ssh_timeout,
    )
    if check and proc.returncode != 0:
        raise RuntimeError(
            f"remote command failed rc={proc.returncode}\n"
            f"--- script ---\n{script}\n--- output ---\n{proc.stdout}"
        )
    return proc


def http_json(url: str, payload: dict | None, timeout: float) -> dict:
    if payload is None:
        req = urllib.request.Request(url, method="GET")
    else:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            method="POST",
            headers={"Content-Type": "application/json"},
        )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8", errors="replace")
    return json.loads(raw)


def is_ready(args: argparse.Namespace) -> bool:
    try:
        http_json(args.base_url.rstrip("/") + "/models", None, timeout=2.0)
        return True
    except (TimeoutError, urllib.error.URLError, json.JSONDecodeError):
        return False


def wait_ready(args: argparse.Namespace, label: str, log_path: str) -> None:
    deadline = time.time() + args.ready_timeout
    while time.time() < deadline:
        if is_ready(args):
            print_event("server_ready", label=label)
            return
        time.sleep(1.0)
    log = ssh(args, f"tail -120 {shlex.quote(log_path)} 2>/dev/null || true\n",
              check=False).stdout
    raise TimeoutError(
        f"{label} did not become ready within {args.ready_timeout:.1f}s\n{log}"
    )


def stop_remote_server(args: argparse.Namespace) -> None:
    """Stop ds4-server on the target port and wait until it is really gone."""
    script = f"""
set -u
tmux kill-session -t {shlex.quote(args.tmux_session)} 2>/dev/null || true
tmux kill-session -t {shlex.quote(args.final_session)} 2>/dev/null || true
pids="$(pgrep -f 'ds4-server .*--port {args.port}' 2>/dev/null || true)"
if [ -n "$pids" ]; then
    kill $pids 2>/dev/null || true
fi
deadline=$((SECONDS + {int(args.shutdown_timeout)}))
while [ "$SECONDS" -lt "$deadline" ]; do
    pids="$(pgrep -f 'ds4-server .*--port {args.port}' 2>/dev/null || true)"
    listeners="$(ss -ltnp 2>/dev/null | grep ':{args.port} ' || true)"
    if [ -z "$pids" ] && [ -z "$listeners" ]; then
        exit 0
    fi
    sleep 1
done
pids="$(pgrep -f 'ds4-server .*--port {args.port}' 2>/dev/null || true)"
if [ -n "$pids" ]; then
    kill -KILL $pids 2>/dev/null || true
fi
deadline=$((SECONDS + 10))
while [ "$SECONDS" -lt "$deadline" ]; do
    pids="$(pgrep -f 'ds4-server .*--port {args.port}' 2>/dev/null || true)"
    listeners="$(ss -ltnp 2>/dev/null | grep ':{args.port} ' || true)"
    if [ -z "$pids" ] && [ -z "$listeners" ]; then
        exit 0
    fi
    sleep 1
done
echo "failed to stop ds4-server on port {args.port}" >&2
pgrep -af 'ds4-server' >&2 || true
ss -ltnp 2>/dev/null | grep ':{args.port} ' >&2 || true
exit 1
"""
    ssh(args, script, timeout=args.shutdown_timeout + 20)
    print_event("server_stopped", port=args.port)


def server_command(args: argparse.Namespace, variant: Variant) -> list[str]:
    cmd = [
        "./ds4-server",
        "--host", args.host,
        "--port", str(args.port),
        "--ctx", str(args.ctx),
        "--threads", str(args.threads),
        "--model", args.remote_model,
        "--tokens", str(args.server_tokens),
        "--cuda",
        "--cpu-moe",
        "--cors",
    ]
    if variant.mode == "mtp":
        draft = variant.mtp_draft if variant.mtp_draft is not None else args.mtp_draft
        cmd.extend(["--mtp", args.remote_mtp_model, "--mtp-draft", str(draft)])
        if variant.mtp_margin is not None:
            cmd.extend(["--mtp-margin", f"{variant.mtp_margin:g}"])
        if variant.mtp_gpu:
            cmd.append("--mtp-gpu")
        if variant.mtp_cpu:
            cmd.append("--mtp-cpu")
    return cmd


def start_remote_server(args: argparse.Namespace, variant: Variant, *,
                        final: bool = False) -> str:
    if variant.mode not in {"mtp", "nomtp"}:
        raise ValueError(variant.mode)
    stop_remote_server(args)
    label = f"{variant.label}{'-final' if final else ''}"
    session = args.final_session if final else args.tmux_session
    log_path = f"{args.remote_log_dir.rstrip('/')}/ds4-{label}.log"
    remote_env = dict(DEFAULT_REMOTE_ENV)
    if variant.strict:
        remote_env["DS4_MTP_STRICT"] = "1"
    env_prefix = " ".join(
        f"{key}={shlex.quote(value)}" for key, value in remote_env.items()
    )
    cmd = " ".join(shlex.quote(part) for part in server_command(args, variant))
    inner = (
        f"cd {shlex.quote(args.remote_build_dir)} && "
        f"exec env {env_prefix} {cmd} > {shlex.quote(log_path)} 2>&1"
    )
    script = f"""
set -e
tmux new-session -d -s {shlex.quote(session)} -- bash -lc {shlex.quote(inner)}
sleep 1
if ! tmux has-session -t {shlex.quote(session)} 2>/dev/null; then
    tail -120 {shlex.quote(log_path)} 2>/dev/null || true
    exit 1
fi
"""
    ssh(args, script)
    print_event("server_start", mode=variant.label, final=final, log=log_path)
    wait_ready(args, label, log_path)
    return log_path


def make_request(args: argparse.Namespace) -> dict:
    return {
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": args.prompt}],
        "think": False,
        "thinking": {"type": "disabled"},
        "max_tokens": args.max_tokens,
        "temperature": 0,
        "stream": False,
    }


def response_text(resp: dict) -> str:
    choice = (resp.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    return str(msg.get("content") or choice.get("text") or "")


def run_request(args: argparse.Namespace, variant: Variant, run_label: str) -> Sample:
    payload = make_request(args)
    started = time.time()
    resp = http_json(
        args.base_url.rstrip("/") + "/chat/completions",
        payload,
        timeout=args.request_timeout,
    )
    seconds = time.time() - started
    usage = resp.get("usage") or {}
    tokens = usage.get("completion_tokens")
    text = response_text(resp)
    if not isinstance(tokens, int) or tokens <= 0:
        tokens = max(1, len(str(text).split()))
    finish = str((resp.get("choices") or [{}])[0].get("finish_reason"))
    output_hash = hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:16]
    sample = Sample(tokens=tokens, seconds=seconds, tok_s=tokens / seconds,
                    finish_reason=finish, output_hash=output_hash)
    print_event(
        "sample",
        mode=variant.label,
        run=run_label,
        completion_tokens=sample.tokens,
        seconds=round(sample.seconds, 3),
        tok_s=round(sample.tok_s, 3),
        finish_reason=sample.finish_reason,
        output_hash=sample.output_hash,
    )
    return sample


def parse_mtp_stats_line(line: str) -> dict[str, int]:
    return {match.group(1): int(match.group(2)) for match in MTP_STATS_RE.finditer(line)}


def sum_mtp_stats(rows: list[dict[str, int]]) -> dict[str, int]:
    keys = sorted({key for row in rows for key in row})
    return {key: sum(row.get(key, 0) for row in rows) for key in keys}


def read_measured_mtp_stats(args: argparse.Namespace, log_path: str,
                            measured_runs: int) -> dict[str, int]:
    if measured_runs <= 0:
        return {}
    script = f"grep ' mtp attempts=' {shlex.quote(log_path)} 2>/dev/null || true\n"
    out = ssh(args, script, check=False).stdout
    rows = [parse_mtp_stats_line(line) for line in out.splitlines() if " mtp attempts=" in line]
    rows = [row for row in rows if row]
    if not rows:
        return {}
    return sum_mtp_stats(rows[-measured_runs:])


def bench_variant(args: argparse.Namespace, variant: Variant) -> tuple[list[Sample], dict[str, int]]:
    log_path = start_remote_server(args, variant)
    for idx in range(args.warmups):
        run_request(args, variant, f"warmup-{idx + 1}")
    samples = [run_request(args, variant, str(idx + 1)) for idx in range(args.runs)]
    rates = [sample.tok_s for sample in samples]
    mtp_stats = read_measured_mtp_stats(args, log_path, args.runs)
    print_event(
        "summary",
        mode=variant.label,
        mean_tok_s=round(statistics.mean(rates), 3),
        median_tok_s=round(statistics.median(rates), 3),
        rates=[round(rate, 3) for rate in rates],
        output_hashes=[sample.output_hash for sample in samples],
        mtp=mtp_stats,
    )
    return samples, mtp_stats


def main() -> int:
    args = parse_args()
    variants = selected_variants(args)
    results: dict[str, list[Sample]] = {}
    mtp_results: dict[str, dict[str, int]] = {}
    restore_error: Exception | None = None
    try:
        for variant in variants:
            samples, mtp_stats = bench_variant(args, variant)
            results[variant.label] = samples
            mtp_results[variant.label] = mtp_stats
        baseline = statistics.mean(sample.tok_s for sample in results["nomtp"]) if "nomtp" in results else None
        baseline_hashes = [sample.output_hash for sample in results["nomtp"]] if "nomtp" in results else None
        for label, samples in results.items():
            mean_tps = statistics.mean(sample.tok_s for sample in samples)
            values: dict[str, object] = {
                "mode": label,
                "mean_tok_s": round(mean_tps, 3),
                "median_tok_s": round(statistics.median(sample.tok_s for sample in samples), 3),
                "output_hashes": [sample.output_hash for sample in samples],
                "mtp": mtp_results.get(label, {}),
            }
            if baseline:
                values["speedup_vs_nomtp"] = round(mean_tps / baseline, 4)
                values["delta_pct_vs_nomtp"] = round((mean_tps / baseline - 1.0) * 100.0, 2)
            if baseline_hashes is not None:
                hashes = [sample.output_hash for sample in samples]
                values["same_output_hashes_as_nomtp"] = hashes == baseline_hashes
            print_event("verdict", **values)
    finally:
        try:
            if args.restore == "none":
                stop_remote_server(args)
            else:
                if args.restore == "mtp":
                    restore_variant = VARIANTS["mtp-fast-m3"]
                elif args.restore == "mtp-cpu":
                    restore_variant = VARIANTS["mtp-fast-m3-cpu"]
                else:
                    restore_variant = VARIANTS["nomtp"]
                if restore_variant.mode == "mtp":
                    restore_variant = Variant(
                        label=args.restore,
                        mode="mtp",
                        mtp_draft=args.mtp_draft,
                        mtp_margin=restore_variant.mtp_margin,
                        strict=restore_variant.strict,
                        mtp_gpu=restore_variant.mtp_gpu,
                        mtp_cpu=restore_variant.mtp_cpu,
                    )
                start_remote_server(args, restore_variant, final=True)
                print_event("restored", mode=args.restore)
        except Exception as exc:  # Report after the primary benchmark error.
            restore_error = exc
            print_event("restore_failed", error=str(exc))
    if restore_error is not None and not results:
        raise restore_error
    return 0


if __name__ == "__main__":
    sys.exit(main())
