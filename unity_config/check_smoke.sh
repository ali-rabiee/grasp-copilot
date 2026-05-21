#!/usr/bin/env bash
# check_smoke.sh — sanity-check a noise-sweep output directory
#
# Bundles five smoke-test checks into one command. Run after a sweep
# finishes (smoke or full) to verify it looks healthy before downstream use.
#
# Usage:
#   bash unity_config/check_smoke.sh                              # auto-pick newest sweep
#   bash unity_config/check_smoke.sh <sweep_dir>                  # explicit path
#   bash unity_config/check_smoke.sh -k oracle_woz_lora           # by key
#
# Examples:
#   bash unity_config/check_smoke.sh                              # latest sweep under user_input_noise/sweeps/
#   bash unity_config/check_smoke.sh -k oracle_woz_lora
#   bash unity_config/check_smoke.sh evaluation/results/robustness/user_input_noise/sweeps/oracle_woz_lora
#
# Exit codes:
#   0 = all checks passed
#   1 = at least one CHECK or FAIL
#   2 = usage error
#
# Checks:
#   (1) Sweep directory + meta JSON exist
#   (2) rollouts.csv exists and is non-trivial
#   (3) Row count is sensible (header + at least ~20 rollouts)
#   (4) Per-(mode, condition, difficulty) sanity:
#         manual rows: success_rate >= 0.95
#         prime  rows: success_rate >= 0.60, mean_motion_tool_calls >= 1, max_ticks_rate <= 0.30
#   (5) Slurm log (if present) shows no Python tracebacks or CUDA OOM
#
# Manual-only sweeps automatically skip the prime-row checks in (4).

set -uo pipefail

SWEEPS_ROOT="evaluation/results/robustness/user_input_noise/sweeps"

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

# Auto-pick newest sweep if none given.
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

echo "── checking sweep: $SWEEP_DIR ─────────────────"
echo

n_fail=0
n_warn=0
n_ok=0

_ok()    { printf "  \033[32mOK\033[0m    %s\n" "$1"; n_ok=$((n_ok+1)); }
_warn()  { printf "  \033[33mWARN\033[0m  %s\n" "$1"; n_warn=$((n_warn+1)); }
_check() { printf "  \033[33mCHECK\033[0m %s\n" "$1"; n_fail=$((n_fail+1)); }
_fail()  { printf "  \033[31mFAIL\033[0m  %s\n" "$1"; n_fail=$((n_fail+1)); }

# ── 1) directory + meta ───────────────────────────────────────────────────

echo "[1] directory + sweep_meta.json"
META="$SWEEP_DIR/sweep_meta.json"
if [ -f "$META" ]; then
    _ok "sweep_meta.json present"
    elapsed=$(python3 -c "import json; print(json.load(open('$META')).get('elapsed_sec','?'))" 2>/dev/null || echo '?')
    total=$(python3 -c "import json; print(json.load(open('$META')).get('total_rollouts','?'))" 2>/dev/null || echo '?')
    backend=$(python3 -c "import json; print(json.load(open('$META')).get('config',{}).get('backend','?'))" 2>/dev/null || echo '?')
    echo "          elapsed=${elapsed}s  total_rollouts=${total}  backend=${backend}"
else
    _check "sweep_meta.json missing (job may have died before writing it)"
fi
echo

# ── 2) rollouts.csv exists + non-trivial ─────────────────────────────────

echo "[2] rollouts.csv presence + size"
ROLL="$SWEEP_DIR/rollouts.csv"
if [ ! -f "$ROLL" ]; then
    _fail "rollouts.csv missing — sweep may have died before any rollout completed"
elif [ ! -s "$ROLL" ]; then
    _fail "rollouts.csv is empty"
else
    bytes=$(wc -c < "$ROLL")
    _ok  "rollouts.csv present (${bytes} bytes)"
fi
echo

# ── 3) row count sensible ─────────────────────────────────────────────────

echo "[3] rollout count"
if [ -f "$ROLL" ]; then
    rows=$(($(wc -l < "$ROLL") - 1))   # subtract header
    if [ "$rows" -lt 1 ]; then
        _fail "rollouts.csv has no data rows"
    elif [ "$rows" -lt 20 ]; then
        _warn "rollouts.csv has only $rows rows — may be a tiny smoke; check it ran what you expected"
    else
        _ok "rollouts.csv has $rows rows"
    fi

    # Per-mode count breakdown.
    python3 - "$ROLL" <<'PY' || _fail "could not parse rollouts.csv"
import csv, sys
from collections import Counter
with open(sys.argv[1]) as f:
    rows = list(csv.DictReader(f))
c = Counter(r['mode'] for r in rows)
print('          by mode: ' + ', '.join(f'{k}={v}' for k, v in sorted(c.items())))
PY
else
    _check "(skipped: no rollouts.csv)"
fi
echo

# ── 4) by_condition sanity ───────────────────────────────────────────────

echo "[4] by_condition.csv sanity"
BC="$SWEEP_DIR/by_condition.csv"
if [ ! -f "$BC" ]; then
    _fail "by_condition.csv missing — aggregator did not run"
else
    python3 - "$BC" <<'PY'
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1])))
any_prime = any(r['mode'] == 'prime' for r in rows)

def show(tag, msg):
    color = {'OK': '\033[32m', 'WARN': '\033[33m', 'CHECK': '\033[33m', 'FAIL': '\033[31m'}.get(tag, '')
    reset = '\033[0m'
    print(f"  {color}{tag}{reset}  {msg}")

problems = 0
for r in rows:
    mode = r['mode']; cond = r['condition']; diff = r['difficulty']
    sr = float(r['success_rate'])
    mtc = float(r['mean_motion_tool_calls'])
    mxr = float(r.get('max_ticks_rate') or 0.0)
    inputs = float(r['mean_total_inputs'])

    label = f"{mode:6s} {cond:13s} {diff:5s}  succ={sr:.2f}  inputs={inputs:5.1f}  motion={mtc:4.1f}  maxticks={mxr:.2f}"

    if mode == 'manual':
        if sr >= 0.95:
            show('OK', label)
        elif sr >= 0.80:
            show('WARN', label + '  (success < 95% on manual is unusual)')
            problems += 1
        else:
            show('FAIL', label + '  (manual mode should always succeed at clean)')
            problems += 1
    elif mode == 'prime':
        cond_msgs = []
        if sr < 0.60:        cond_msgs.append('low success')
        if mtc < 1:          cond_msgs.append('LLM never emitted a motion tool')
        if mxr > 0.30:       cond_msgs.append('too many max_ticks failures')
        if not cond_msgs:
            show('OK', label)
        else:
            show('CHECK', label + '  ← ' + '; '.join(cond_msgs))
            problems += 1

if not any_prime:
    print("  (manual-only sweep — no PRIME rows to evaluate)")

sys.exit(1 if problems else 0)
PY
    rc=$?
    if [ "$rc" -ne 0 ]; then n_fail=$((n_fail+1)); fi
fi
echo

# ── 5) slurm / stdout log scan ───────────────────────────────────────────

echo "[5] log scan for errors"
LOG_DIR="logs/slurm"
LOG_PATTERN="${LOG_DIR}/*${SWEEP_NAME}*.log"
shopt -s nullglob
LOGS=( $LOG_PATTERN )
shopt -u nullglob
if [ ${#LOGS[@]} -eq 0 ]; then
    _warn "no slurm log matching '${LOG_PATTERN}' (ok if you ran locally without --output)"
else
    LOG="${LOGS[-1]}"
    bad=$(grep -E "Traceback|CUDA out of memory|ModuleNotFoundError|FileNotFoundError|Killed|RuntimeError|FATAL" "$LOG" | head -5 || true)
    if [ -n "$bad" ]; then
        _check "log contains error-looking lines (showing first 5):"
        echo "$bad" | sed 's/^/        /'
    else
        _ok "log clean (no Traceback / OOM / Killed / FATAL)"
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
    echo "→ Smoke passes but with warnings — inspect output before the long run."
    exit 0
fi
echo
echo "→ All checks passed. Cleared to launch the full sweep."
