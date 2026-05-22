#!/usr/bin/env bash
set -euo pipefail

# Full-stack DS4 hybrid profiling harness.
#
# This launches ds4-server under Nsight Systems, drives the server-shaped decode
# fixture against it, and captures sidecar CPU/GPU/PCIe/memory-pressure counters
# over the same request window.

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$ROOT_DIR"

STAMP="$(date +%Y%m%d-%H%M%S)"
OUTDIR="${OUTDIR:-/tmp/ds4-profile-${STAMP}}"
MODEL_PATH="${MODEL_PATH:-${ROOT_DIR}/ds4flash.gguf}"
PROMPT_FILE="${PROMPT_FILE:-${ROOT_DIR}/long_memory_archive.txt}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-18080}"
CTX="${CTX:-8192}"
THREADS="${THREADS:-16}"
MAX_TOKENS="${MAX_TOKENS:-64}"
REQUESTS="${REQUESTS:-2}"
TIMEOUT="${TIMEOUT:-1200}"
SERVER_BIN="${SERVER_BIN:-${ROOT_DIR}/ds4-server}"
BENCH_SCRIPT="${BENCH_SCRIPT:-${ROOT_DIR}/scripts/bench_server_decode_cache.py}"
MODEL_NAME="${MODEL_NAME:-deepseek-chat}"
MONITOR_SECONDS="${MONITOR_SECONDS:-1800}"
NSYS_TRACE="${NSYS_TRACE:-cuda,nvtx,osrt}"
NSYS_SAMPLE="${NSYS_SAMPLE:-process-tree}"
NSYS_BACKTRACE="${NSYS_BACKTRACE:-lbr}"
NSYS_CUDA_MEMORY_USAGE="${NSYS_CUDA_MEMORY_USAGE:-false}"

mkdir -p "$OUTDIR"

SERVER_LOG="$OUTDIR/ds4-server.log"
BENCH_LOG="$OUTDIR/bench.jsonl"
ENV_LOG="$OUTDIR/env.txt"
PID_LOG="$OUTDIR/pids.txt"
SUMMARY="$OUTDIR/summary.txt"

SERVER_PID=""
NSYS_PID=""
MONITOR_PIDS=()

log() {
    printf '[profile] %s\n' "$*" >&2
}

cleanup() {
    local pid
    for pid in "${MONITOR_PIDS[@]:-}"; do
        stop_monitor_pid "$pid"
    done
    if [ -n "${SERVER_PID:-}" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
        kill -INT "$SERVER_PID" 2>/dev/null || true
    fi
    if [ -n "${NSYS_PID:-}" ] && kill -0 "$NSYS_PID" 2>/dev/null; then
        wait "$NSYS_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

require_file() {
    if [ ! -f "$1" ]; then
        printf 'missing required file: %s\n' "$1" >&2
        exit 1
    fi
}

require_file "$MODEL_PATH"
require_file "$PROMPT_FILE"
require_file "$SERVER_BIN"
require_file "$BENCH_SCRIPT"

export DS4_CPU_AFFINITY="${DS4_CPU_AFFINITY:-1}"
export DS4_CPU_AFFINITY_LIST="${DS4_CPU_AFFINITY_LIST:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}"
export DS4_CPU_MOE_ROW_CHUNK="${DS4_CPU_MOE_ROW_CHUNK:-8}"
export DS4_CPU_MOE_DECODE_TIMING="${DS4_CPU_MOE_DECODE_TIMING:-1}"
export DS4_CPU_MOE_TAIL_PROFILE="${DS4_CPU_MOE_TAIL_PROFILE:-1}"
export DS4_CUDA_DIRECT_MODEL="${DS4_CUDA_DIRECT_MODEL:-1}"
export DS4_CUDA_PARTIAL_WEIGHT_CACHE="${DS4_CUDA_PARTIAL_WEIGHT_CACHE:-1}"
export DS4_CUDA_WEIGHT_CACHE_LIMIT_GB="${DS4_CUDA_WEIGHT_CACHE_LIMIT_GB:-10}"
export DS4_CUDA_REQUIRE_DENSE_WEIGHT_CACHE="${DS4_CUDA_REQUIRE_DENSE_WEIGHT_CACHE:-1}"
export DS4_CUDA_DYNAMIC_EXPERTS="${DS4_CUDA_DYNAMIC_EXPERTS:-1}"
export DS4_CUDA_DYNAMIC_EXPERT_CACHE_GB="${DS4_CUDA_DYNAMIC_EXPERT_CACHE_GB:-4}"
export DS4_CUDA_DYNAMIC_EXPERT_POLICY="${DS4_CUDA_DYNAMIC_EXPERT_POLICY:-lru}"
export DS4_CUDA_DYNAMIC_EXPERT_MAX_EVICTIONS="${DS4_CUDA_DYNAMIC_EXPERT_MAX_EVICTIONS:-64}"
export DS4_CUDA_DYNAMIC_EXPERT_RESERVE_MB="${DS4_CUDA_DYNAMIC_EXPERT_RESERVE_MB:-512}"
export DS4_CUDA_DYNAMIC_EXPERT_GROUP_SIZE="${DS4_CUDA_DYNAMIC_EXPERT_GROUP_SIZE:-4}"
export DS4_CUDA_DYNAMIC_EXPERT_GROUP_EVICTION="${DS4_CUDA_DYNAMIC_EXPERT_GROUP_EVICTION:-1}"
export DS4_CUDA_DYNAMIC_EXPERT_DECODE_EAGER="${DS4_CUDA_DYNAMIC_EXPERT_DECODE_EAGER:-1}"
export DS4_CUDA_DYNAMIC_EXPERT_DECODE_EVICT="${DS4_CUDA_DYNAMIC_EXPERT_DECODE_EVICT:-0}"
export DS4_CUDA_DYNAMIC_EXPERT_DECODE_MIN_SCORE="${DS4_CUDA_DYNAMIC_EXPERT_DECODE_MIN_SCORE:-8}"
export DS4_CUDA_DYNAMIC_EXPERT_DECODE_MAINTENANCE_INTERVAL="${DS4_CUDA_DYNAMIC_EXPERT_DECODE_MAINTENANCE_INTERVAL:-1}"
export DS4_CUDA_DYNAMIC_EXPERT_DECODE_MAX_PROMOTIONS="${DS4_CUDA_DYNAMIC_EXPERT_DECODE_MAX_PROMOTIONS:-4}"
export DS4_CUDA_DYNAMIC_EXPERT_DECODE_GROUP_SIZE="${DS4_CUDA_DYNAMIC_EXPERT_DECODE_GROUP_SIZE:-4}"
export DS4_CUDA_LAYERWISE_PREFILL_STAGING="${DS4_CUDA_LAYERWISE_PREFILL_STAGING:-1}"
export DS4_CUDA_LAYERWISE_PREFILL_STAGING_STICKY="${DS4_CUDA_LAYERWISE_PREFILL_STAGING_STICKY:-1}"
export DS4_CUDA_LAYERWISE_PREFILL_STAGING_OVERLAP="${DS4_CUDA_LAYERWISE_PREFILL_STAGING_OVERLAP:-1}"
export DS4_CUDA_LAYERWISE_PREFILL_STAGING_MB="${DS4_CUDA_LAYERWISE_PREFILL_STAGING_MB:-512}"
export DS4_CUDA_LAYERWISE_PREFILL_STAGING_MIN_PAIRS="${DS4_CUDA_LAYERWISE_PREFILL_STAGING_MIN_PAIRS:-16}"
export DS4_CUDA_LAYERWISE_PREFILL_STAGING_MAX_EXPERTS="${DS4_CUDA_LAYERWISE_PREFILL_STAGING_MAX_EXPERTS:-2}"
export DS4_CUDA_ROUTE_PROFILE="${DS4_CUDA_ROUTE_PROFILE:-${OUTDIR}/route-profile.tsv}"

{
    date --iso-8601=ns
    printf 'ROOT_DIR=%s\n' "$ROOT_DIR"
    printf 'OUTDIR=%s\n' "$OUTDIR"
    printf 'MODEL_PATH=%s\n' "$MODEL_PATH"
    printf 'PROMPT_FILE=%s\n' "$PROMPT_FILE"
    printf 'HOST=%s\nPORT=%s\nCTX=%s\nTHREADS=%s\nMAX_TOKENS=%s\nREQUESTS=%s\n' \
        "$HOST" "$PORT" "$CTX" "$THREADS" "$MAX_TOKENS" "$REQUESTS"
    env | sort | grep -E '^(DS4_|CUDA_|NVIDIA_|OMP_|OPENBLAS_|BLIS_)' || true
    printf '\n--- lscpu ---\n'
    lscpu || true
    printf '\n--- nvidia-smi ---\n'
    nvidia-smi --query-gpu=name,driver_version,pci.bus_id,pcie.link.gen.current,pcie.link.width.current,memory.total --format=csv || true
} > "$ENV_LOG" 2>&1

wait_ready() {
    python3 - "$HOST" "$PORT" "$TIMEOUT" <<'PY'
import json
import sys
import time
import urllib.request

host, port, timeout = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
deadline = time.time() + timeout
url = f"http://{host}:{port}/v1/models"
while time.time() < deadline:
    try:
        with urllib.request.urlopen(url, timeout=2.0) as resp:
            json.loads(resp.read().decode("utf-8", errors="replace"))
        sys.exit(0)
    except Exception:
        time.sleep(0.5)
print(f"server did not become ready: {url}", file=sys.stderr)
sys.exit(1)
PY
}

find_server_pid() {
    local deadline=$((SECONDS + 60))
    local pid=""
    while [ "$SECONDS" -lt "$deadline" ]; do
        pid="$(pgrep -P "${NSYS_PID}" ds4-server 2>/dev/null | head -1 || true)"
        if [ -z "$pid" ]; then
            pid="$(pgrep -f "ds4-server .*--port ${PORT}" 2>/dev/null | head -1 || true)"
        fi
        if [ -n "$pid" ]; then
            printf '%s\n' "$pid"
            return 0
        fi
        sleep 0.5
    done
    return 1
}

start_monitor() {
    setsid "$@" &
    MONITOR_PIDS+=("$!")
}

stop_monitor_pid() {
    local pid="${1:-}"
    if [ -z "$pid" ]; then
        return
    fi
    # Monitors that go through sudo can leave a root-owned child alive if only
    # the sudo wrapper is signaled. Start every monitor in its own process group
    # and tear down the whole group; use sudo as a fallback for root children.
    kill -TERM -- "-$pid" 2>/dev/null || true
    sudo -n kill -TERM -- "-$pid" 2>/dev/null || true
    kill -TERM "$pid" 2>/dev/null || true
}

valid_perf_events() {
    local ev
    for ev in "$@"; do
        if sudo -n perf stat -e "$ev" -- sleep 0.01 >/dev/null 2>&1; then
            printf '%s\n' "$ev"
        fi
    done
}

start_sidecars() {
    local valid_events
    mapfile -t valid_events < <(valid_perf_events \
        cycles \
        instructions \
        cache-references \
        cache-misses \
        branches \
        branch-misses \
        stalled-cycles-frontend \
        ls_any_fills_from_sys.dram_io_near \
        ls_dmnd_fills_from_sys.dram_io_near \
        ls_hw_pf_dc_fills.dram_io_near)

    if command -v nvidia-smi >/dev/null 2>&1; then
        start_monitor nvidia-smi dmon -s pucmt -d 1 -o TD --format csv,nounit -f "$OUTDIR/nvidia-dmon.csv"
        start_monitor nvidia-smi pmon -s um -d 1 -o TD -f "$OUTDIR/nvidia-pmon.log"
    fi
    if command -v vmstat >/dev/null 2>&1; then
        setsid vmstat -t 1 > "$OUTDIR/vmstat.log" 2>&1 &
        MONITOR_PIDS+=("$!")
    fi
    if command -v turbostat >/dev/null 2>&1; then
        setsid sudo -n turbostat --quiet --show Busy%,Bzy_MHz,PkgWatt --interval 1 \
            > "$OUTDIR/turbostat.log" 2> "$OUTDIR/turbostat.err" &
        MONITOR_PIDS+=("$!")
    fi
    if [ "${#valid_events[@]}" -gt 0 ]; then
        local events
        events="$(IFS=,; printf '%s' "${valid_events[*]}")"
        printf '%s\n' "$events" > "$OUTDIR/perf-events.txt"
        setsid sudo -n perf stat -a -I 1000 -x, -o "$OUTDIR/perf-stat.csv" -e "$events" -- sleep "$MONITOR_SECONDS" &
        MONITOR_PIDS+=("$!")
    fi
    if [ -n "$SERVER_PID" ]; then
        (
            while kill -0 "$SERVER_PID" 2>/dev/null; do
                date --iso-8601=ns
                ps -L -p "$SERVER_PID" -o pid,tid,psr,pcpu,stat,comm --no-headers 2>/dev/null || true
                sleep 1
            done
        ) > "$OUTDIR/thread-ps.log" 2>&1 &
        MONITOR_PIDS+=("$!")
        (
            while kill -0 "$SERVER_PID" 2>/dev/null; do
                printf '%s\n' "$(date --iso-8601=ns)"
                grep -E '^(VmRSS|VmHWM|VmData|VmStk|VmExe|VmLib|VmPTE|voluntary_ctxt_switches|nonvoluntary_ctxt_switches):' \
                    "/proc/${SERVER_PID}/status" 2>/dev/null || true
                cat "/proc/${SERVER_PID}/io" 2>/dev/null || true
                printf '\n'
                sleep 1
            done
        ) > "$OUTDIR/proc-memory-io.log" 2>&1 &
        MONITOR_PIDS+=("$!")
    fi
}

log "starting ds4-server under Nsight Systems"
nsys profile \
    --force-overwrite=true \
    --trace="$NSYS_TRACE" \
    --sample="$NSYS_SAMPLE" \
    --cpuctxsw=process-tree \
    --backtrace="$NSYS_BACKTRACE" \
    --cuda-memory-usage="$NSYS_CUDA_MEMORY_USAGE" \
    --export=sqlite \
    -o "$OUTDIR/nsys-ds4-server" \
    "$SERVER_BIN" \
        --host "$HOST" \
        --port "$PORT" \
        --ctx "$CTX" \
        --threads "$THREADS" \
        --model "$MODEL_PATH" \
        --tokens "$MAX_TOKENS" \
        --cuda \
        --cpu-moe \
        --chdir "$ROOT_DIR" \
        > "$SERVER_LOG" 2>&1 &
NSYS_PID="$!"

wait_ready
SERVER_PID="$(find_server_pid)"
{
    printf 'nsys_pid=%s\n' "$NSYS_PID"
    printf 'server_pid=%s\n' "$SERVER_PID"
} > "$PID_LOG"

log "server ready; starting sidecar profilers"
start_sidecars
date --iso-8601=ns > "$OUTDIR/request-window-start.txt"

log "running server-shaped decode fixture"
python3 "$BENCH_SCRIPT" \
    --no-start-server \
    --model-path "$MODEL_PATH" \
    --prompt-file "$PROMPT_FILE" \
    --host "$HOST" \
    --port "$PORT" \
    --ctx "$CTX" \
    --threads "$THREADS" \
    --max-tokens "$MAX_TOKENS" \
    --requests "$REQUESTS" \
    --timeout "$TIMEOUT" \
    --model-name "$MODEL_NAME" \
    > "$BENCH_LOG" 2>&1

date --iso-8601=ns > "$OUTDIR/request-window-end.txt"

log "stopping server so Nsight can finalize"
kill -INT "$SERVER_PID" 2>/dev/null || true
wait "$NSYS_PID" 2>/dev/null || true
NSYS_PID=""
SERVER_PID=""

for pid in "${MONITOR_PIDS[@]:-}"; do
    stop_monitor_pid "$pid"
done
wait 2>/dev/null || true
MONITOR_PIDS=()

if command -v nsys >/dev/null 2>&1 && [ -f "$OUTDIR/nsys-ds4-server.nsys-rep" ]; then
    nsys stats "$OUTDIR/nsys-ds4-server.nsys-rep" > "$OUTDIR/nsys-stats.txt" 2> "$OUTDIR/nsys-stats.err" || true
fi

python3 - "$OUTDIR" > "$SUMMARY" <<'PY'
import csv
import datetime
import json
import re
import statistics
import sys
from pathlib import Path

out = Path(sys.argv[1])

def parse_iso8601_ns(text):
    text = text.strip().replace(",", ".", 1)
    m = re.match(r"(.*\.)(\d{6})\d*([+-]\d\d:\d\d)$", text)
    if m:
        text = m.group(1) + m.group(2) + m.group(3)
    return datetime.datetime.fromisoformat(text)

window_start = None
window_end = None
window_s = None
try:
    window_start = parse_iso8601_ns((out / "request-window-start.txt").read_text())
    window_end = parse_iso8601_ns((out / "request-window-end.txt").read_text())
    window_s = (window_end - window_start).total_seconds()
except Exception:
    pass

print(f"outdir: {out}")
if window_s is not None:
    print(f"request_window_s: {window_s:.3f}")
print()

bench = out / "bench.jsonl"
if bench.exists():
    rows = []
    for line in bench.read_text(errors="replace").splitlines():
        try:
            obj = json.loads(line)
        except Exception:
            continue
        if obj.get("event") == "request":
            rows.append(obj)
    if rows:
        print("requests:")
        for row in rows:
            toks = row.get("completion_tokens") or 0
            elapsed = row.get("elapsed_s") or 0
            tps = toks / elapsed if toks and elapsed else 0.0
            print(f"  request {row.get('request')}: {elapsed:.3f}s, completion={toks}, end_to_end~{tps:.2f} tok/s")
        print()

server = out / "ds4-server.log"
if server.exists():
    interesting = []
    pats = (
        "prefill:",
        "CUDA hybrid prefill timing:",
        "CUDA hybrid decode timing:",
        "CUDA dynamic experts:",
        "CUDA layerwise prefill staging:",
        "CPU-MoE decode timing:",
        "CPU-MoE selected timing:",
        "CPU-MoE tail profile",
    )
    for line in server.read_text(errors="replace").splitlines():
        if any(p in line for p in pats):
            interesting.append(line)
    if interesting:
        print("server counters:")
        for line in interesting[-24:]:
            print(f"  {line}")
        print()

dmon = out / "nvidia-dmon.csv"
if dmon.exists():
    rows = []
    with dmon.open(newline="", errors="replace") as f:
        reader = csv.reader(f)
        headers = None
        for row in reader:
            if not row:
                continue
            row = [c.strip().lstrip("#") for c in row]
            if row[0] == "Date":
                headers = row
                continue
            if headers and len(row) == len(headers):
                rec = dict(zip(headers, row))
                if window_start is not None and window_end is not None:
                    try:
                        dt = datetime.datetime.strptime(
                            rec["Date"] + " " + rec["Time"],
                            "%Y%m%d %H:%M:%S",
                        ).replace(tzinfo=window_start.tzinfo)
                        if not (window_start <= dt <= window_end):
                            continue
                    except Exception:
                        pass
                rows.append(rec)
    def nums(name):
        vals = []
        for row in rows:
            try:
                vals.append(float(row.get(name, "").replace("-", "nan")))
            except Exception:
                pass
        return [v for v in vals if v == v]
    if rows:
        print("nvidia dmon request-window averages/max:")
        for name in ("sm", "mem", "pwr", "fb", "rxpci", "txpci"):
            vals = nums(name)
            if vals:
                print(f"  {name}: avg={statistics.fmean(vals):.2f}, max={max(vals):.2f}")
        print("  note: rxpci/txpci are NVIDIA dmon PCIe throughput samples, usually MB/s with nounit formatting.")
        print()

perf = out / "perf-stat.csv"
if perf.exists():
    text = perf.read_text(errors="replace")
    totals = {}
    for line in text.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            continue
        try:
            ts = float(parts[0])
            val = float(parts[1])
        except Exception:
            continue
        if window_s is not None and ts > window_s + 2.0:
            continue
        event = parts[3]
        totals[event] = totals.get(event, 0.0) + val
    if totals:
        print("perf request-window totals:")
        for event in sorted(totals):
            val = totals[event]
            if "dram_io" in event:
                gib = val * 64.0 / (1024.0 ** 3)
                suffix = f", ~={gib / window_s:.2f} GiB/s" if window_s else ""
                print(f"  {event}: {val:.0f} fills ~= {gib:.2f} GiB of 64B lines{suffix}")
            else:
                print(f"  {event}: {val:.0f}")
        print()

nsys_stats = out / "nsys-stats.txt"
if nsys_stats.exists():
    text = nsys_stats.read_text(errors="replace")
    sections = []
    for heading in ("CUDA API Statistics", "CUDA GPU Kernel Summary", "CUDA GPU MemOps Summary", "OS Runtime Summary"):
        idx = text.find(heading)
        if idx >= 0:
            sections.append(text[idx:idx+1800].rstrip())
    if sections:
        print("nsys stats excerpts:")
        for section in sections:
            print(section)
            print()

print("artifacts:")
for name in (
    "nsys-ds4-server.nsys-rep",
    "nsys-ds4-server.sqlite",
    "nsys-stats.txt",
    "bench.jsonl",
    "ds4-server.log",
    "nvidia-dmon.csv",
    "nvidia-pmon.log",
    "perf-stat.csv",
    "vmstat.log",
    "turbostat.log",
    "thread-ps.log",
    "proc-memory-io.log",
    "route-profile.tsv",
):
    p = out / name
    if p.exists():
        print(f"  {p}")
PY

cat "$SUMMARY"
log "profile complete: $OUTDIR"
