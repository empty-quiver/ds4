#!/usr/bin/env python3
"""Server-shaped DS4 decode/cache benchmark fixture.

The fixture launches ds4-server by default, sends several related stateless
chat-completion requests with a large shared prompt prefix, and reports per
request latency/output stats.  It intentionally keeps the server process alive
across requests so CUDA expert residency, sticky staging, and KV prefix reuse
can behave like an agent/server workload instead of a one-shot CLI run.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--binary", default="./ds4-server")
    p.add_argument("--model-path", required=True)
    p.add_argument("--prompt-file", required=True)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=18080)
    p.add_argument("--ctx", type=int, default=4096)
    p.add_argument("--threads", type=int, default=16)
    p.add_argument("--max-tokens", type=int, default=192)
    p.add_argument("--requests", type=int, default=4)
    p.add_argument("--timeout", type=float, default=900.0)
    p.add_argument("--server-log", default="/tmp/ds4-server-decode-cache.log")
    p.add_argument("--kv-disk-dir", default=None,
                   help="Optional ds4-server KV cache directory for prefix reuse.")
    p.add_argument("--kv-disk-space-mb", type=int, default=4096)
    p.add_argument("--no-start-server", action="store_true",
                   help="Use an already-running server at --host/--port.")
    p.add_argument("--server-arg", action="append", default=[],
                   help="Extra ds4-server argument. Repeat for each argument.")
    p.add_argument("--model-name", default="deepseek-chat",
                   help="Request model. deepseek-chat disables thinking.")
    p.add_argument("--print-output", action="store_true")
    return p.parse_args()


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


def wait_ready(base_url: str, proc: subprocess.Popen | None, timeout: float) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc is not None and proc.poll() is not None:
            raise RuntimeError(f"server exited early with status {proc.returncode}")
        try:
            http_json(base_url + "/v1/models", None, timeout=2.0)
            return
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
            time.sleep(0.5)
    raise TimeoutError(f"server did not become ready within {timeout:.1f}s")


def launch_server(args: argparse.Namespace) -> tuple[subprocess.Popen | None, object | None]:
    if args.no_start_server:
        return None, None
    log = open(args.server_log, "w", encoding="utf-8")
    cmd = [
        args.binary,
        "--host", args.host,
        "--port", str(args.port),
        "--ctx", str(args.ctx),
        "--threads", str(args.threads),
        "--model", args.model_path,
        "--tokens", str(args.max_tokens),
        "--cuda",
        "--cpu-moe",
        "--chdir", str(Path(args.binary).resolve().parent),
    ]
    if args.kv_disk_dir:
        cmd.extend([
            "--kv-disk-dir", args.kv_disk_dir,
            "--kv-disk-space-mb", str(args.kv_disk_space_mb),
        ])
    cmd.extend(args.server_arg)
    print(json.dumps({"event": "server_start", "cmd": cmd, "log": args.server_log}),
          flush=True)
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
    return proc, log


def build_request(base_prompt: str, args: argparse.Namespace, idx: int) -> dict:
    # Keep almost all text identical across requests so server prefix reuse and
    # retained expert boxes get a realistic stateless-agent shape.
    task = (
        f"\n\nDecode cache fixture request {idx + 1} of {args.requests}.\n"
        "Write exactly 24 numbered lines. Each line must start with "
        f"R{idx + 1:02d}-LNN, where NN is the two digit line number. "
        "Each line should mention the archive, a component, an observation, "
        "and a consequence. Do not summarize. Do not stop before line 24."
    )
    return {
        "model": args.model_name,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a deterministic benchmark writer. Follow the "
                    "requested output length exactly and avoid early stops."
                ),
            },
            {"role": "user", "content": base_prompt + task},
        ],
        "temperature": 0,
        "top_p": 1,
        "min_p": 0,
        "max_tokens": args.max_tokens,
        "stream": False,
    }


def extract_text(resp: dict) -> str:
    choices = resp.get("choices") or []
    if not choices:
        return ""
    msg = choices[0].get("message") or {}
    content = msg.get("content")
    return content if isinstance(content, str) else ""


def main() -> int:
    args = parse_args()
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except AttributeError:
        pass
    base_url = f"http://{args.host}:{args.port}"
    base_prompt = Path(args.prompt_file).read_text(encoding="utf-8")
    env_snapshot = {
        k: v for k, v in sorted(os.environ.items())
        if k.startswith("DS4_CUDA_DYNAMIC_EXPERT") or
           k.startswith("DS4_CUDA_LAYERWISE_PREFILL") or
           k.startswith("DS4_CUDA_HOT_EXPERT")
    }
    print(json.dumps({"event": "env", "values": env_snapshot}), flush=True)

    proc = None
    log = None
    rows = []
    try:
        proc, log = launch_server(args)
        wait_ready(base_url, proc, args.timeout)
        for i in range(args.requests):
            payload = build_request(base_prompt, args, i)
            t0 = time.time()
            resp = http_json(base_url + "/v1/chat/completions", payload, args.timeout)
            elapsed = time.time() - t0
            text = extract_text(resp)
            usage = resp.get("usage") or {}
            row = {
                "request": i + 1,
                "elapsed_s": round(elapsed, 3),
                "chars": len(text),
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "total_tokens": usage.get("total_tokens"),
                "finish_reason": (resp.get("choices") or [{}])[0].get("finish_reason"),
            }
            rows.append(row)
            print(json.dumps({"event": "request", **row}), flush=True)
            if args.print_output:
                print(text)
        print(json.dumps({"event": "summary", "rows": rows}), flush=True)
        return 0
    finally:
        if proc is not None and proc.poll() is None:
            proc.send_signal(signal.SIGINT)
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.terminate()
                proc.wait(timeout=10)
        if log is not None:
            log.close()
        if not args.no_start_server and args.server_log:
            print(json.dumps({"event": "server_log", "path": args.server_log}), flush=True)


if __name__ == "__main__":
    sys.exit(main())
