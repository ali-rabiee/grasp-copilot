#!/usr/bin/env bash
# check_reranker.sh — sanity-check a reranker ablation sweep output
#
# Mirrors check_smoke.sh's structure (5 checks) and adds a 6th: dialogs.jsonl
# parses + IG values are in [0, log2(n_candidates_before)].
#
# Usage:
#   bash unity_config/check_reranker.sh                                  # newest sweep
#   bash unity_config/check_reranker.sh <sweep_dir>
#   bash unity_config/check_reranker.sh -k oracle_woz_lora__info_gain
#
# Exit codes:
#   0 = all checks passed
#   1 = at least one CHECK or FAIL
#   2 = usage error

set -uo pipefail

SWEEPS_ROOT="evaluation/results/reranker/ablation"

usage() {
    grep -E '^#( |$)' "$0" | sed 's/^# \?//'
    exit 2
}

SWEEP_DIR=""
while [ $# -gt 0 ]; do
    case "$1" in
        -k|--key)   shift; SWEEP_DIR="$SWEEPS_ROOT/$1"; shift ;;
        -h|--help)  usage ;;
        -*)         echo "unknown flag: $1" >&2; usage ;;
        *)          SWEEP_DIR="$1"; shift ;;
    esac
done

if [ -z "$SWEEP_DIR" ]; then
    if [ ! -d "$SWEEPS_ROOT" ]; then
        echo "FAIL  no sweeps root: $SWEEPS_ROOT" >&2
        exit 1
    fi
    SWEEP_DIR=$(ls -td "$SWEEPS_ROOT"/*/ 2>/dev/null | head -n1)
    SWEEP_DIR="${SWEEP_DIR%/}"
    if [ -z "$SWEEP_DIR" ]; then
        echo "FAIL  no sweep subdirs under $SWEEPS_ROOT" >&2
        exit 1
    fi
fi

if [ ! -d "$SWEEP_DIR" ]; then
    echo "FAIL  sweep dir not found: $SWEEP_DIR" >&2
    exit 1
fi

SWEEP_NAME=$(basename "$SWEEP_DIR")
echo "── checking reranker sweep: $SWEEP_DIR ─────────────────"
echo

n_fail=0
n_warn=0
n_ok=0

_ok()    { printf "  \033[32mOK\033[0m    %s\n" "$1"; n_ok=$((n_ok+1)); }
_warn()  { printf "  \033[33mWARN\033[0m  %s\n" "$1"; n_warn=$((n_warn+1)); }
_check() { printf "  \033[33mCHECK\033[0m %s\n" "$1"; n_fail=$((n_fail+1)); }
_fail()  { printf "  \033[31mFAIL\033[0m  %s\n" "$1"; n_fail=$((n_fail+1)); }

# ── 1) sweep_meta.json ────────────────────────────────────────────────────

echo "[1] sweep_meta.json + rerank config"
META="$SWEEP_DIR/sweep_meta.json"
if [ -f "$META" ]; then
    _ok "sweep_meta.json present"
    mode=$(python3 -c "import json; print(json.load(open('$META')).get('rerank',{}).get('mode','?'))" 2>/dev/null || echo '?')
    k=$(python3 -c "import json; print(json.load(open('$META')).get('rerank',{}).get('k','?'))" 2>/dev/null || echo '?')
    backend=$(python3 -c "import json; print(json.load(open('$META')).get('config',{}).get('backend','?'))" 2>/dev/null || echo '?')
    total=$(python3 -c "import json; print(json.load(open('$META')).get('total_rollouts','?'))" 2>/dev/null || echo '?')
    echo "          backend=${backend}  rerank_mode=${mode}  k=${k}  total_rollouts=${total}"
else
    _check "sweep_meta.json missing (job may have died before writing it)"
fi
echo

# ── 2) rollouts.csv ───────────────────────────────────────────────────────

echo "[2] rollouts.csv presence + size"
ROLL="$SWEEP_DIR/rollouts.csv"
if [ ! -f "$ROLL" ]; then
    _fail "rollouts.csv missing"
elif [ ! -s "$ROLL" ]; then
    _fail "rollouts.csv is empty"
else
    bytes=$(wc -c < "$ROLL")
    rows=$(($(wc -l < "$ROLL") - 1))
    _ok "rollouts.csv present (${bytes} bytes, ${rows} rows)"
fi
echo

# ── 3) by_condition.csv prime sanity ─────────────────────────────────────

echo "[3] by_condition.csv prime sanity"
BC="$SWEEP_DIR/by_condition.csv"
if [ ! -f "$BC" ]; then
    _fail "by_condition.csv missing"
else
    python3 - "$BC" <<'PY'
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1])))
prime = [r for r in rows if r['mode'] == 'prime']
def show(tag, msg):
    color = {'OK':'\033[32m','WARN':'\033[33m','CHECK':'\033[33m','FAIL':'\033[31m'}.get(tag,'')
    print(f"  {color}{tag}\033[0m  {msg}")
problems = 0
for r in prime:
    sr  = float(r['success_rate'])
    mtc = float(r['mean_motion_tool_calls'])
    mxr = float(r.get('max_ticks_rate') or 0.0)
    label = f"prime {r['condition']:13s} {r['difficulty']:5s}  succ={sr:.2f}  motion={mtc:4.1f}  maxticks={mxr:.2f}"
    msgs = []
    if sr < 0.50:        msgs.append('low success')
    if mtc < 1:          msgs.append('no motion tools')
    if mxr > 0.40:       msgs.append('too many max_ticks')
    if not msgs:
        show('OK', label)
    else:
        show('CHECK', label + '  ← ' + '; '.join(msgs))
        problems += 1
sys.exit(1 if problems else 0)
PY
    rc=$?
    if [ "$rc" -ne 0 ]; then n_fail=$((n_fail+1)); fi
fi
echo

# ── 4) dialogs.jsonl exists + parses ─────────────────────────────────────

echo "[4] dialogs.jsonl presence + parse"
DLG="$SWEEP_DIR/dialogs.jsonl"
if [ ! -f "$DLG" ]; then
    _warn "dialogs.jsonl missing (sweep ran with rerank_mode=none and dialog_log disabled?)"
else
    n_lines=$(wc -l < "$DLG")
    if [ "$n_lines" -lt 1 ]; then
        _check "dialogs.jsonl is empty"
    else
        _ok "dialogs.jsonl present ($n_lines lines)"
    fi
fi
echo

# ── 5) IG range sanity ────────────────────────────────────────────────────

echo "[5] IG range sanity (0 ≤ ig_chosen ≤ log2(n_candidates_before))"
if [ -f "$DLG" ]; then
    python3 - "$DLG" <<'PY'
import json, math, sys
bad = 0
n = 0
mean_ig = 0.0
frac_pos = 0
with open(sys.argv[1]) as f:
    for line in f:
        line = line.strip()
        if not line: continue
        try: rec = json.loads(line)
        except Exception:
            bad += 1; continue
        n += 1
        ig = float(rec.get('ig_bits', 0))
        n_b = int(rec.get('n_candidates_before', 0))
        cap = math.log2(max(n_b, 1)) + 1e-6
        if ig < -1e-6 or ig > cap + 1e-3:
            bad += 1
        mean_ig += ig
        if ig > 0: frac_pos += 1
mean_ig = mean_ig / max(n, 1)
frac_pos = frac_pos / max(n, 1)
print(f"          n_dialogs={n}  mean_IG={mean_ig:.3f} bits  frac_IG>0={frac_pos:.2f}  bad_rows={bad}")
sys.exit(1 if bad else 0)
PY
    rc=$?
    if [ "$rc" -ne 0 ]; then _check "IG sanity check failed (see above)"; else _ok "IG values within bounds"; fi
fi
echo

# ── 6) slurm log scan ────────────────────────────────────────────────────

echo "[6] log scan for errors"
LOG_DIR="logs/slurm"
LOG_PATTERN="${LOG_DIR}/*${SWEEP_NAME}*.log"
shopt -s nullglob
LOGS=( $LOG_PATTERN )
shopt -u nullglob
if [ ${#LOGS[@]} -eq 0 ]; then
    _warn "no slurm log matching '${LOG_PATTERN}'"
else
    LOG="${LOGS[-1]}"
    bad=$(grep -E "Traceback|CUDA out of memory|ModuleNotFoundError|FileNotFoundError|Killed|RuntimeError|FATAL|No space left|CANCELLED|DUE TO TIME LIMIT|slurmstepd: error|oom-kill" "$LOG" | head -10 || true)
    if [ -n "$bad" ]; then
        _check "log contains error-looking lines (first 10):"
        echo "$bad" | sed 's/^/        /'
    else
        _ok "log clean"
    fi
    last=$(tail -5 "$LOG" 2>/dev/null || true)
    if [ -n "$last" ]; then
        echo "        — last 5 log lines —"
        echo "$last" | sed 's/^/        /'
    fi
fi
echo

# ── summary ──────────────────────────────────────────────────────────────

echo "── summary ─────────────────────────────────────"
printf "  OK: %d   WARN: %d   FAIL/CHECK: %d\n" "$n_ok" "$n_warn" "$n_fail"
if [ "$n_fail" -gt 0 ]; then
    echo
    echo "→ NOT clear to launch the full run yet. Address CHECK/FAIL items above."
    exit 1
fi
if [ "$n_warn" -gt 0 ]; then
    echo
    echo "→ Smoke passes with warnings — inspect output before the long run."
    exit 0
fi
echo
echo "→ All checks passed. Cleared to launch the full sweep."
