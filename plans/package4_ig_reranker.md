# Package 4 · Information-Gain Reranker — Implementation Plan

**Project:** PRIME IROS 2026 extension — add a post-hoc information-gain reranker that selects *which* question to ask among K LLM-generated candidates, and produces an interpretability story showing PRIME's questions are quantifiably informative.

**Status:** Drafted 2026-05-20. Authoritative until the headline ablation table in §7 is filled in.

**Owner:** Ali Rabiee. Depends on artifacts from `plans/training_woz_envs.md` (trained LoRAs) and `plans/noise-from-real-data.md` (scenario corpus + rollout loop).

---

## 0 · TL;DR

- **Reranker = post-LLM filter, no retraining.** When the policy decides INTERACT, generate K=5 candidate questions at temperature 0.7, score each by expected entropy reduction over the candidate set, pick argmax.
- **Does NOT touch ask-vs-act.** WoZ policy still owns that decision. Reranker only chooses *what* to ask among questions the policy already wants to ask.
- **Two evaluation surfaces, two sbatch jobs:**
  - `job_reranker_ablation.sbatch` — online rollouts in the `scenario_noise_sweep` matrix → headline efficiency/success claims.
  - `job_ig_analysis.sbatch` — post-hoc IG analysis on logged dialogs → distribution plot + comparisons to random / oracle / no-rerank.
- **Headline ablation matrix:** {WoZ-only, WoZ+IG-rerank, Oracle+IG-rerank} × full noise sweep on `oracle_woz_lora` and `oracle_lora`. ~3 sbatch submissions, ~60h GPU.
- **Validation gates** (from the package brief):
  1. Reranker implemented as a post-LLM filter, zero retraining.
  2. WoZ+rerank either improves task efficiency OR matches WoZ-only with measurably more informative questions.
  3. PRIME questions show mean IG ≥ 0.5 bits.

---

## 1 · Strategic decisions and their justification

### 1.1 Inference-time reranker, not training signal → **post-LLM filter only**

The package brief is explicit. Folding IG into training would (a) require labeled "good question" supervision we don't have, (b) confound with the WoZ contribution, and (c) make the claim "WoZ teaches informative questions" non-falsifiable. As a post-LLM filter, the reranker is a clean ablation knob — flip it on/off, attribute the delta.

### 1.2 K=5 LLM samples, temperature 0.7 → **matches the brief literally**

Confirmed via discussion 2026-05-20. K=5 candidate generations per ASK tick means ~5× the per-INTERACT LLM cost. With ~3 ASKs/episode and ~9,600 rollouts/cell, that is bearable on a single GPU within the 20h SBATCH budget — see compute estimate in §6.

Templated candidates (full-menu, binary-confirm) were rejected as auxiliaries; the paper's claim is "we rerank the LLM's own ideas," and mixing templated candidates dilutes it.

### 1.3 Pruning logic source → **lift from `data_generator/oracle.py`**

The oracle's reply handler (`OracleBackend.on_user_reply` in `evaluation/benchmarks/scenario_noise_sweep.py`, lines 134–233) already implements deterministic candidate-set updates per response, mirroring `oracle.py`'s context types (`confirm`, `candidate_choice`, `anything_else`, `mode_select`, `intent_gate_*`). Reuse this logic verbatim inside the reranker's simulator — do **not** write a second pruning rule set. Any drift between the oracle's pruning and the reranker's pruning would produce uninterpretable IG numbers.

### 1.4 Three-condition online headline + offline-only controls → **decided 2026-05-20**

Online sweep covers {WoZ-only, WoZ+rerank, Oracle+rerank}. Offline IG analysis adds random-question and oracle-question controls *on the same logged INTERACT calls* — much cheaper than running a 4th/5th rollout matrix, and the IG metric is well-defined offline.

---

## 2 · Code organization (the "clean subfolder names" the user asked for)

```
llm/
└── reranker/                       # NEW — the post-LLM filter module
    ├── __init__.py                 # public: `make_reranked_backend`, `RerankerConfig`
    ├── candidates.py               # K-sample candidate-question generation
    ├── pruning.py                  # simulate `update_state(reply) → candidates`
    ├── entropy.py                  # H(C), priors (uniform | motion-weighted)
    ├── selector.py                 # info_gain / random / oracle / none policies
    ├── policy_wrapper.py           # wraps any backend(input)->call; intercepts INTERACT
    └── tests/
        ├── test_pruning_parity.py  # parity vs OracleBackend.on_user_reply
        ├── test_entropy.py         # closed-form: H(uniform-2)=1, H(uniform-4)=2
        └── test_passthrough.py     # mode="none" trajectory == bare backend

evaluation/
├── reranker/                       # NEW — sweep runner + post-hoc analysis
│   ├── __init__.py
│   ├── dialog_logger.py            # JSONL writer for every emitted INTERACT
│   ├── run_reranker_sweep.py       # thin shim over scenario_noise_sweep with --rerank_mode
│   ├── analyze_ig.py               # IG distribution, by interact_type, comparisons
│   ├── plot_ig.py                  # paper figures
│   └── tables_reranker.py          # LaTeX/CSV: headline ablation table
└── results/
    └── reranker/                   # NEW — outputs land here
        ├── ablation/<model>__<rerank_mode>/
        │   ├── rollouts.csv
        │   ├── by_condition.csv
        │   ├── dialogs.jsonl       # one row per emitted INTERACT (for IG analysis)
        │   └── sweep_meta.json
        ├── ig_analysis/
        │   ├── per_question.csv    # one row per logged INTERACT × selector
        │   ├── ig_distribution.{pdf,png}
        │   ├── ig_by_kind.{pdf,png}
        │   └── summary.json
        └── tables/
            ├── table_reranker_ablation.{csv,tex}
            └── table_ig_summary.{csv,tex}

unity_config/
├── job_reranker_ablation.sbatch    # NEW — online sweep, one (model, rerank_mode) per job
├── job_ig_analysis.sbatch          # NEW — post-hoc analysis (CPU only)
└── check_reranker.sh               # NEW — smoke validator mirroring check_smoke.sh

plans/
└── package4_ig_reranker.md         # this file (renamed from `package4_IG -Reranker`)
```

**Why this layout:**
- `llm/reranker/` lives next to `llm/inference.py` because it consumes the same `_load_model_and_tokenizer` / `_generate_once` helpers. One Python import, no cross-package coupling.
- `evaluation/reranker/` mirrors `evaluation/benchmarks/` so the sbatch templates and result locations follow the conventions already used by `run_paper_benchmark.py` and `scenario_noise_sweep.py`.
- Results land under `evaluation/results/reranker/` to keep the existing `paper_benchmark/` and `robustness/` trees untouched.

---

## 3 · Reranker module — concrete file contracts

### 3.1 `llm/reranker/candidates.py`

```python
@dataclass(frozen=True)
class CandidateQuestion:
    tool_call: Dict       # full tool-call dict (must have tool=="INTERACT")
    raw_text: str         # LLM raw output (for debug)
    sample_idx: int       # 0..K-1
    log_prob: Optional[float] = None  # if available

def generate_candidates(
    input_dict: Dict,
    *, model, tok, base_cfg: InferenceConfig,
    k: int = 5, temperature: float = 0.7, top_p: float = 0.95,
    seed: int,
) -> List[CandidateQuestion]:
    """K samples at temperature>0; filtered to those whose top-level tool=="INTERACT".
    If <K pass the filter, return whatever passed (the selector handles len=1)."""
```

Reuses the patched `_load_model_and_tokenizer` / `_generate_once` from `llm/inference.py` (don't re-import the bug-workaround block; it's already applied at module load). The only new dependency is the per-call temperature override for sampling.

### 3.2 `llm/reranker/pruning.py`

```python
@dataclass
class PruneSnapshot:
    candidates: List[str]           # candidate obj_ids
    excluded_obj_ids: List[str]
    awaiting: Dict[str, bool]       # awaiting_* flags from OracleState
    state_summary: str              # cheap hash for caching

def simulate_reply(
    state_before: PruneSnapshot,
    tool_call: Dict,                # the INTERACT call being scored
    reply_choice_idx: int,          # 0..len(choices)-1
    objects: Sequence[Dict],
    memory: Dict,
) -> PruneSnapshot:
    """Returns the candidate snapshot AFTER applying the reply.
    Lifted verbatim from OracleBackend.on_user_reply in
    evaluation/benchmarks/scenario_noise_sweep.py — same context-type
    handling (confirm, candidate_choice, anything_else, mode_select,
    intent_gate_*). MUST stay byte-identical to the oracle's pruning."""
```

**Acceptance criterion:** for every (state, reply) pair in `data/woz_phase2/llm_contract_valid.jsonl`, the snapshot produced by `simulate_reply` matches the snapshot produced by running the actual oracle. The parity test in §10 step 1 enforces this.

### 3.3 `llm/reranker/entropy.py`

```python
def entropy_bits(candidates: Sequence[str], *, prior: Mapping[str, float] | None = None) -> float:
    """Shannon entropy in bits. Uniform prior if `prior` is None."""

def motion_weighted_prior(
    candidates: Sequence[str],
    gripper_hist: Sequence[Dict],
    objects: Sequence[Dict],
) -> Dict[str, float]:
    """Inverse-distance softmax weighted by recent gripper motion direction.
    Same formulation as the SA1 baseline (offline_exec_benchmark.py:_heuristic_predict_then_assist),
    so the prior is grounded in an existing baseline rather than invented here."""

def expected_post_entropy(
    candidate_question: Dict,           # the INTERACT tool_call
    state_before: PruneSnapshot,
    *, prior_over_replies: Sequence[float],
    pruning_fn: Callable,               # simulate_reply
    objects, memory,
) -> Tuple[float, List[Tuple[int, float, float]]]:
    """Returns (E_r[H(C|r)], per-reply [(idx, P(r), H_after)])."""
```

Two priors over user replies are supported:
- **Uniform** (default; matches the brief).
- **Motion-weighted**: reweight YES/NO and per-object choices by the candidate-set posterior from §3.3, so "the user is more likely to confirm the closest object" matches the SA2 baseline.

The IG of a candidate question is then `IG = H(C_before) − E_r[H(C|r)]`.

### 3.4 `llm/reranker/selector.py`

```python
class Selector(Protocol):
    name: str
    def pick(self, candidates: List[CandidateQuestion], scored: List[float]) -> int: ...

class InfoGainSelector:      # argmax IG, tie-break by smaller len(choices)
class RandomSelector:        # uniform random among candidates (control)
class OracleSelector:        # picks the candidate closest in JSON to oracle's emit
class NoneSelector:          # always picks candidates[0] (= bare LLM output)
```

A single `make_selector(name: str) -> Selector` factory keeps the sweep runner config simple.

### 3.5 `llm/reranker/policy_wrapper.py`

```python
def make_reranked_backend(
    inner_backend: Callable[[Dict], Optional[Dict]],   # the WoZ/Oracle LLM backend
    *, model, tok, base_cfg, k: int, temperature: float,
    selector: Selector,
    dialog_log: Optional[DialogLogger] = None,
) -> Callable[[Dict], Optional[Dict]]:
    """Returns a new backend with the same callable signature.

    Flow per call:
      1. raw = inner_backend(input)              # one shot — decides ask-vs-act
      2. if raw is None or raw["tool"] != "INTERACT": return raw   # passthrough
      3. cands = generate_candidates(...)        # K samples (raw is treated as #0 if it parses)
      4. for each cand: score = IG(cand)         # via pruning + entropy
      5. pick  = selector.pick(cands, scores)
      6. if dialog_log: log H_before, all per-candidate IGs, the choice
      7. return cands[pick].tool_call
    """
```

**Critical guarantee:** if `inner_backend` returns a motion tool (APPROACH / ALIGN_YAW / STACK / GRAB / POUR) or `None`, the wrapped backend returns it unchanged. This enforces validation criterion #1 ("does not change ask-vs-act").

### 3.6 `llm/reranker/__init__.py`

Re-exports `make_reranked_backend`, `RerankerConfig`, the selector factory, and `entropy_bits` for the analysis script. Nothing else.

---

## 4 · Sweep runner — `evaluation/reranker/run_reranker_sweep.py`

A thin shim over `evaluation/benchmarks/scenario_noise_sweep.py`, **not** a copy. Imports the existing `_build_hf_backend`, then wraps it:

```python
def main():
    # ... same CLI as scenario_noise_sweep, plus:
    ap.add_argument("--rerank_mode", default="info_gain",
                    choices=["none", "info_gain", "random", "oracle"])
    ap.add_argument("--k_candidates", type=int, default=5)
    ap.add_argument("--rerank_temperature", type=float, default=0.7)
    ap.add_argument("--prior", default="uniform", choices=["uniform", "motion_weighted"])
    ap.add_argument("--dialog_log", default=None,
                    help="Path to dialogs.jsonl; if set, every emitted INTERACT is logged.")
```

`out_dir/<model>__<rerank_mode>/` contains:
- `rollouts.csv`, `by_condition.csv`, `sweep_meta.json` (same as the existing sweep).
- `dialogs.jsonl` — one line per emitted INTERACT call across all rollouts. Schema:

```json
{"scenario_id": "...", "seed": 0, "condition": "clean", "tick": 14,
 "state_summary_hash": "...",
 "h_before_bits": 2.32, "h_after_expected_bits": 1.05, "ig_bits": 1.27,
 "selector": "info_gain", "k_candidates": 5,
 "chosen": {"tool":"INTERACT", "args":{...}},
 "candidates": [{"args":..., "ig_bits": 1.27, "per_reply": [[0, 0.5, 0.0], [1, 0.5, 2.10]]}, ...],
 "interact_kind": "QUESTION", "context_type": "candidate_choice",
 "n_candidates_before": 5}
```

This single JSONL is the input to **all** offline analyses (§5), so we never have to re-run rollouts to add a new IG metric — just re-process `dialogs.jsonl`.

---

## 5 · Post-hoc IG analysis — `evaluation/reranker/analyze_ig.py`

Inputs:
- `dialogs.jsonl` from each sweep cell (read all under `evaluation/results/reranker/ablation/`).

For every logged INTERACT call, recompute four scores using the **same** `entropy.py` / `pruning.py`:

| Score | What it represents |
|---|---|
| `ig_chosen` | The IG of the selector's pick (already in the JSONL). |
| `ig_random` | Expected IG of a uniformly-random candidate (mean over the K logged candidates). |
| `ig_oracle` | IG of the candidate closest in JSON to the oracle's emit at the same `state_summary_hash` (run oracle once per unique state). |
| `ig_no_rerank` | IG of the LLM's first-sample candidate (= what WoZ would have asked without reranker). |

Outputs (under `evaluation/results/reranker/ig_analysis/`):

- `per_question.csv` — one row per (dialog × selector).
- `ig_distribution.{pdf,png}` — overlapping histograms of IG distributions across selectors.
- `ig_by_kind.{pdf,png}` — facet plot: IG distribution per `interact_kind ∈ {QUESTION, SUGGESTION, CONFIRM}`.
- `summary.json`:
  ```json
  {"mean_ig_chosen": 1.04, "mean_ig_no_rerank": 0.83,
   "mean_ig_random": 0.21, "mean_ig_oracle": 1.12,
   "frac_chosen_ge_0.5bits": 0.71,
   "kind_breakdown": {"QUESTION": {...}, "CONFIRM": {...}, "SUGGESTION": {...}}}
  ```

The `summary.json` is what `tables_reranker.py` consumes to fill `table_ig_summary.tex`.

---

## 6 · Compute budget

| Job | Backend cost | Per-cell wallclock | Cells | Total |
|---|---|---|---|---|
| Online sweep: `oracle_woz_lora` + IG rerank | LLM × 5 per INTERACT (~0.5 LLM calls / tick avg) | ~18h | 1 | ~18h GPU |
| Online sweep: `oracle_woz_lora` baseline (no rerank) | LLM × 1 per tick | ~14h | 1 | ~14h GPU |
| Online sweep: `oracle_lora` + IG rerank | same as #1 | ~18h | 1 | ~18h GPU |
| Online sweep: WoZ random + WoZ oracle rerank (controls) | deferred — recomputed offline | — | 0 | 0h GPU |
| Offline IG analysis | CPU | ~10 min | 1 | trivial |
| **Total** | | | | **~50h GPU + minutes CPU** |

The three online sweeps fit within Unity's 20h-per-SLURM-job budget as single submissions and run in parallel on different nodes.

If timing slips, drop K from 5 → 3 (cuts ~40% off the IG cost; the brief allows it).

---

## 7 · The headline ablation table (what goes in the paper)

`table_reranker_ablation.tex` — produced by `evaluation/reranker/tables_reranker.py`:

| Model | Rerank | Success% | Mean #interactions | Mean completion (s) | Mean IG (bits) | Frac IG≥0.5 |
|---|---|---|---|---|---|---|
| Oracle (heuristic, ref) | — | (sanity) | — | — | — | — |
| WoZ (`oracle_woz_lora`) | — | … | … | … | … | … |
| WoZ + IG-rerank | info_gain | **bold if best** | … | … | … | … |
| WoZ (offline-only) | random | — | — | — | … | … |
| WoZ (offline-only) | oracle | — | — | — | … | … |
| Oracle-only (`oracle_lora`) + IG-rerank | info_gain | … | … | … | … | … |

Three flavors of row:
- **Online rollout cells** — Success/Interactions/Time come from `rollouts.csv`; Mean IG comes from `dialogs.jsonl`.
- **Offline-only cells** — Success/Interactions/Time are blank (N/A); only Mean IG is reported (post-hoc replay of the WoZ online dialogs through the random/oracle selector).
- **Per-condition splits** — every cell has a `clean` and `compound_mid` variant so the noise story isn't lost in the average.

---

## 8 · Paper claims this pipeline can defend

| # | Claim | Backed by |
|---|---|---|
| A | "PRIME's INTERACT prompts deliver mean information gain of **X bits** per question (≥ 0.5 bits per the brief)." | `ig_distribution.pdf` + `summary.json:mean_ig_chosen` |
| B | "The IG reranker reduces mean #interactions-to-completion by Δ% on `oracle_woz_lora` at matched success rate." | `table_reranker_ablation.tex`, rows 2 vs 3 |
| C | "WoZ training already produces informative questions: WoZ-only mean IG = 0.83 bits ≫ random 0.21 bits; the reranker's lift is +X bits — the reranker validates the WoZ signal rather than replacing it." | `summary.json` `mean_ig_no_rerank` vs `mean_ig_random` vs `mean_ig_chosen` |
| D | "The reranker preserves the policy's ask-vs-act distribution — INTERACT rate is statistically indistinguishable between rows 2 and 3." | Per-tick INTERACT counts in `rollouts.csv` (paired t-test by scenario seed) |
| E | "Robustness to user-input noise is preserved with the reranker on." | Per-condition columns of `by_condition.csv`, plotted as a 2-line overlay on top of the existing `robustness_curves.pdf` |
| F (kept honest per brief) | "If WoZ+rerank ≈ WoZ-only on online metrics, this strengthens the WoZ contribution: the policy's questions are already near-optimal under the IG metric." | Same table; the brief mandates not suppressing this. |

---

## 9 · SBATCH templates

### 9.1 `unity_config/job_reranker_ablation.sbatch`

Mirrors `job_noise_sweep.sbatch` exactly. New env vars: `RERANK_MODE`, `K_CANDIDATES`, `RERANK_TEMPERATURE`, `PRIOR`.

```bash
#!/bin/bash
#SBATCH --job-name=rerank_sweep
#SBATCH --partition=gpu,uri-gpu
#SBATCH --gpus=1
#SBATCH --constraint=l40s|a100|h100
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=20:00:00
#SBATCH --open-mode=append

# Submit per (model, rerank_mode) pair. Example launches:
#
#   # Headline:
#   MODEL_KEY=oracle_woz_lora MODEL_PATH=models/qwen2_5_3b_oracle_woz_lora \
#       RERANK_MODE=info_gain \
#       sbatch unity_config/job_reranker_ablation.sbatch
#
#   # Baseline (no reranker — still logs dialogs so analysis can replay):
#   MODEL_KEY=oracle_woz_lora MODEL_PATH=models/qwen2_5_3b_oracle_woz_lora \
#       RERANK_MODE=none \
#       sbatch unity_config/job_reranker_ablation.sbatch
#
#   # Warm-start sanity check:
#   MODEL_KEY=oracle_lora MODEL_PATH=models/qwen2_5_3b_oracle_lora \
#       RERANK_MODE=info_gain \
#       sbatch unity_config/job_reranker_ablation.sbatch

set -euo pipefail
CONDA_ENV="${CONDA_ENV:-copilot}"
MODEL_KEY="${MODEL_KEY:?MODEL_KEY required}"
MODEL_PATH="${MODEL_PATH:?MODEL_PATH required}"
RERANK_MODE="${RERANK_MODE:-info_gain}"
K_CANDIDATES="${K_CANDIDATES:-5}"
RERANK_TEMPERATURE="${RERANK_TEMPERATURE:-0.7}"
PRIOR="${PRIOR:-uniform}"
N_SEEDS="${N_SEEDS:-5}"
MAX_TICKS="${MAX_TICKS:-2000}"
CONDITIONS="${CONDITIONS:-clean dir_low dir_high sel_low sel_high compound_mid}"
MODES="${MODES:-prime}"

SCENARIOS="evaluation/results/robustness/user_input_noise/scenarios/scenarios.labeled.jsonl"
OUT_DIR="evaluation/results/reranker/ablation/${MODEL_KEY}__${RERANK_MODE}"
DIALOG_LOG="$OUT_DIR/dialogs.jsonl"

REPO="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$REPO"

[ -d "$MODEL_PATH" ] || { echo "Model dir not found: $MODEL_PATH"; exit 2; }
[ -f "$SCENARIOS" ] || { echo "Scenarios file not found: $SCENARIOS"; exit 2; }

JOB_NAME="${SLURM_JOB_NAME:-rerank_sweep}"; JOB_ID="${SLURM_JOB_ID:-manual}"
mkdir -p logs/slurm "$OUT_DIR"
exec >>"logs/slurm/${JOB_NAME}_${MODEL_KEY}_${RERANK_MODE}_${JOB_ID}.log" 2>&1

echo "[job] $(date --iso-8601=seconds)  model=$MODEL_KEY  rerank=$RERANK_MODE  k=$K_CANDIDATES"

if [ -f /modules/apps/conda/main/etc/profile.d/conda.sh ]; then
    source /modules/apps/conda/main/etc/profile.d/conda.sh
else
    source "$(conda info --base)/etc/profile.d/conda.sh"
fi
conda activate "$CONDA_ENV"

export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

python -m evaluation.reranker.run_reranker_sweep \
    --scenarios   "$SCENARIOS" \
    --out_dir     "$OUT_DIR" \
    --backend     hf_ft \
    --model_paths "$MODEL_KEY=$MODEL_PATH" \
    --modes       $MODES \
    --conditions  $CONDITIONS \
    --n_seeds     "$N_SEEDS" \
    --max_ticks   "$MAX_TICKS" \
    --rerank_mode "$RERANK_MODE" \
    --k_candidates "$K_CANDIDATES" \
    --rerank_temperature "$RERANK_TEMPERATURE" \
    --prior       "$PRIOR" \
    --dialog_log  "$DIALOG_LOG" \
    --progress_every 100

echo "[job] finished $(date --iso-8601=seconds)"
```

### 9.2 `unity_config/job_ig_analysis.sbatch`

Cheap, CPU-only. Reads all `dialogs.jsonl` under the ablation tree and emits the analysis artifacts + tables.

```bash
#!/bin/bash
#SBATCH --job-name=ig_analysis
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --open-mode=append

set -euo pipefail
CONDA_ENV="${CONDA_ENV:-copilot}"
ABLATION_ROOT="${ABLATION_ROOT:-evaluation/results/reranker/ablation}"
OUT_DIR="${OUT_DIR:-evaluation/results/reranker/ig_analysis}"

REPO="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$REPO"
[ -d "$ABLATION_ROOT" ] || { echo "Ablation root not found: $ABLATION_ROOT"; exit 2; }

JOB_ID="${SLURM_JOB_ID:-manual}"
mkdir -p logs/slurm "$OUT_DIR"
exec >>"logs/slurm/ig_analysis_${JOB_ID}.log" 2>&1

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

python -m evaluation.reranker.analyze_ig \
    --ablation_root "$ABLATION_ROOT" \
    --out_dir       "$OUT_DIR" \
    --include_selectors info_gain random oracle no_rerank

python -m evaluation.reranker.plot_ig --out_dir "$OUT_DIR"
python -m evaluation.reranker.tables_reranker --out_dir evaluation/results/reranker/tables
```

### 9.3 `unity_config/check_reranker.sh`

Mirrors `check_smoke.sh` (5 checks) but adds a 6th: `dialogs.jsonl` exists, every row parses, and `ig_chosen ∈ [0, log2(n_candidates_before)]`.

---

## 10 · Build order (tasks, in dependency order)

1. **`llm/reranker/pruning.py`** + parity test against the oracle on WoZ valid set. Gate: 100% byte-identical snapshots. *(2 days)*
2. **`llm/reranker/entropy.py`** + unit test (closed-form: 2 equiprobable → 1 bit; 4 equiprobable → 2 bits). *(0.5 day)*
3. **`llm/reranker/candidates.py`** + smoke test on a single contract row (assert K=5 returned with `tool=="INTERACT"`). *(0.5 day)*
4. **`llm/reranker/selector.py`** + `policy_wrapper.py` + parity test: `RerankerConfig(mode="none")` produces the same trajectory as the bare backend on a fixed seed. *(1 day)*
5. **`evaluation/reranker/dialog_logger.py`** + `run_reranker_sweep.py` shim. Smoke: 1 model × clean condition × 1 seed, verify `dialogs.jsonl` populates. *(1 day)*
6. **Smoke sbatch** (`N_SEEDS=1 CONDITIONS=clean MAX_TICKS=500`) — confirm one cell completes in <2h on Unity. Run `check_reranker.sh`. *(0.5 day Unity wallclock)*
7. **Headline online runs** — three parallel sbatch submissions per §6. *(~20h Unity wallclock, in parallel)*
8. **`evaluation/reranker/analyze_ig.py`** + `plot_ig.py` + `tables_reranker.py`. *(1.5 days; can start while §7 runs)*
9. **`job_ig_analysis.sbatch`** after §7 completes. *(<1h)*
10. **Paper-ready table + figure pass** — drop into `paper_snippets.tex` next to the existing PRIME tables. *(0.5 day)*

Total: ~7 working days dev + ~1 day Unity wallclock. Fits inside the IROS 2026 submission window assuming `oracle_woz_lora` is already trained per `plans/training_woz_envs.md`.

---

## 11 · Pitfalls to actively avoid

Lifted from the brief and pinned here so they survive implementation:

1. **Do not overclaim.** This is a reranker + post-hoc analysis, not a POMDP solver. Use the phrase "post-hoc reranker" in paper prose; avoid "Bayesian-experimental design system."
2. **Do not let IG override ask-vs-act.** Test #4 (parity with `mode="none"`) is the regression gate. If a future tweak to `policy_wrapper.py` ever changes the per-tick INTERACT distribution, that test must fail.
3. **Do not cherry-pick.** All four selectors (info_gain, random, oracle, no_rerank) are reported in the same table, even if the WoZ-only and WoZ+rerank rows look identical. Per the brief: a null result strengthens the WoZ contribution, it doesn't weaken it.
4. **Do not let pruning drift from the oracle.** `simulate_reply` ≡ `OracleBackend.on_user_reply`. The parity test in step 1 is the only acceptable signal that they match.
5. **Do not skip the IG ≥ 0.5 bits gate.** If `summary.json:mean_ig_chosen < 0.5`, the WoZ model is producing low-signal questions and the paper claim is unsupported. Stop and investigate before writing the paragraph.

---

## 12 · Open questions to resolve before §10 step 6 (smoke submission)

- [ ] Confirm `oracle_woz_lora` is on Unity (per `unity_config/TRANSFER_TO_UNITY_noise_sweep.md` §3). If not, rsync it first.
- [ ] Decide whether the offline analysis should also run against the existing `paper_benchmark` dialogs (the contract JSONLs in `data/woz_phase2/`). This is a free extra data point — same `analyze_ig.py`, different input glob.
- [ ] Pick the SBATCH partition for `job_ig_analysis.sbatch`: Unity has both `cpu` and `cpu-long`; the 1h time fits the short queue.

---

**Footer.** This plan covers Package 4 of the IROS 2026 PRIME extension. Packages 1–3 are tracked in `plans/training_woz_envs.md` and `plans/noise-from-real-data.md`. All three packages share the same trained-model registry, scenario corpus, and SBATCH conventions on Unity.
