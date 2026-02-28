from __future__ import annotations

import argparse
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING, cast

if TYPE_CHECKING:
    # Optional dependency; not required for merged-model inference.
    from peft import PeftModel  # type: ignore[import]  # pragma: no cover

from .utils import json_loads_strict, set_seed
from .data import SYSTEM_PROMPT


@dataclass(frozen=True, slots=True)
class InferenceConfig:
    # Path (local dir or HF id) to a *merged* standalone model.
    model_path: str
    use_4bit: bool = False
    temperature: float = 0.2
    top_p: float = 0.9
    max_new_tokens: int = 512
    seed: int = 0
    deterministic: bool = False


def _resolve_model_path(model_path: str) -> str:
    """
    Resolve a local model directory robustly.

    Why: transformers/huggingface_hub will treat strings like "foo/bar" as a Hub repo id
    if the path doesn't exist on disk. Users often run the GUI from different CWDs.
    """
    p_raw = str(model_path).strip()
    if not p_raw:
        return p_raw

    # 1) As-is (relative to CWD) + user expansion.
    p1 = Path(p_raw).expanduser()
    if p1.exists():
        return str(p1)

    # 2) If the user passed "grasp-copilot/..." while already inside grasp-copilot/,
    # strip the prefix and try again.
    for prefix in ("./grasp-copilot/", "grasp-copilot/"):
        if p_raw.startswith(prefix):
            p2 = Path(p_raw[len(prefix) :]).expanduser()
            if p2.exists():
                return str(p2)

    # 3) Relative to the grasp-copilot package root (repo layout).
    base = Path(__file__).resolve().parents[1]  # .../grasp-copilot
    p3 = (base / p_raw).expanduser()
    if p3.exists():
        return str(p3)

    # 4) If still not found, return the raw string (may be a real HF id).
    return p_raw


def _build_messages(prompt: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]


def _load_model_and_tokenizer(cfg: InferenceConfig):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Workaround: transformers >=5.0 has a bug where fix_mistral_regex is
    # passed both explicitly and via **kwargs in TokenizersBackend.__init__,
    # causing a "multiple values for keyword argument" TypeError.  Popping
    # it before __init__ spreads kwargs prevents the collision.
    try:
        import transformers.tokenization_utils_tokenizers as _tut
        _orig_tb_init = _tut.TokenizersBackend.__init__
        if not getattr(_orig_tb_init, "_fmr_patched", False):
            def _fixed_tb_init(self, *args, **kwargs):
                kwargs.pop("fix_mistral_regex", None)
                return _orig_tb_init(self, *args, **kwargs)
            _fixed_tb_init._fmr_patched = True  # type: ignore[attr-defined]
            _tut.TokenizersBackend.__init__ = _fixed_tb_init
    except Exception:
        pass

    model_path = _resolve_model_path(cfg.model_path)
    try:
        tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=True)
    except TypeError:
        tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
    except Exception as e:
        msg = (
            f"Failed to load tokenizer from '{model_path}'.\n"
            f"- Ensure the model directory contains tokenizer files (e.g., tokenizer.json or vocab.json+merges.txt).\n"
            f"- Ensure optional deps are installed: `pip install tokenizers protobuf`.\n"
            f"Original error: {type(e).__name__}: {e}"
        )
        raise RuntimeError(msg) from e
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    quant_cfg = None
    device_map = "auto" if torch.cuda.is_available() else None
    if cfg.use_4bit:
        # Optional dependency; only required for 4-bit inference.
        try:
            from transformers import BitsAndBytesConfig

            quant_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16 if (torch.cuda.is_available() and getattr(torch.cuda, "is_bf16_supported", lambda: False)()) else torch.float16,
            )
        except Exception as e:
            raise RuntimeError("use_4bit inference requires bitsandbytes + transformers BitsAndBytesConfig") from e

    # Prefer loading directly onto CUDA when available (fast inference in GUI).
    # Some environments may not have `accelerate`, which is needed for device_map="auto".
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            dtype="auto",
            quantization_config=quant_cfg,
            device_map=device_map,
        )
    except Exception:
        # Fallback: load without device_map and move to CUDA if possible.
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            dtype="auto",
            quantization_config=quant_cfg,
        )
        if torch.cuda.is_available():
            # Some transformer stubs confuse type checkers on `.to("cuda")`; cast to Any.
            model = cast(Any, model).to("cuda")

    model.eval()
    return model, tok


def _generate_once(model, tok, messages: List[Dict[str, str]], cfg: InferenceConfig) -> str:
    import torch

    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(text, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        out = model.generate(
            **inputs,
            do_sample=cfg.temperature > 0,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            max_new_tokens=cfg.max_new_tokens,
            pad_token_id=tok.pad_token_id,
        )
    gen = out[0, inputs["input_ids"].shape[-1] :]
    return tok.decode(gen, skip_special_tokens=True).strip()


def generate_json_only(prompt: str, cfg: InferenceConfig) -> Dict[str, Any]:
    set_seed(cfg.seed)
    if cfg.deterministic:
        # Best-effort determinism. Note: some GPU ops may still be nondeterministic depending on
        # hardware/driver/torch version. This is mainly for repeatable debugging.
        try:
            import torch

            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            try:
                torch.use_deterministic_algorithms(True)
            except Exception:
                pass
        except Exception:
            pass
    model, tok = _load_model_and_tokenizer(cfg)

    messages = _build_messages(prompt)
    raw1 = _generate_once(model, tok, messages, cfg)
    try:
        return json_loads_strict(raw1)
    except Exception:
        repair_messages = _build_messages("Return ONLY valid JSON for the previous answer.\n\nPrevious answer:\n" + raw1)
        raw2 = _generate_once(model, tok, repair_messages, cfg)
        try:
            return json_loads_strict(raw2)
        except Exception as e:
            raise ValueError(f"Model did not return valid JSON after repair attempt: {e}\nRAW:\n{raw2}") from e


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", type=str, required=False, default=None, help="Path/HF id of a merged standalone model.")
    # Backward compatible aliases (deprecated).
    ap.add_argument("--model_name", type=str, default=None, help="DEPRECATED: use --model_path")
    ap.add_argument("--merged_model_path", type=str, default=None, help="DEPRECATED: use --model_path")
    ap.add_argument("--adapter_path", type=str, default=None, help="DEPRECATED: adapters are no longer supported here; use merged models.")
    ap.add_argument("--use_4bit", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--prompt", type=str, default=None)
    ap.add_argument("--prompt_file", type=str, default=None)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--deterministic",
        action="store_true",
        help="Best-effort deterministic inference (disables cudnn benchmark, enables deterministic algorithms).",
    )
    args = ap.parse_args()

    # Resolve model path (merged models only).
    model_path = args.model_path or args.merged_model_path or args.model_name
    if not model_path:
        model_path = "Qwen/Qwen2.5-7B-Instruct"
        print("[inference] WARNING: defaulting to Qwen/Qwen2.5-7B-Instruct; pass --model_path for a merged model.")
    if args.adapter_path:
        raise SystemExit("adapter_path is deprecated and not supported. Please pass a merged model via --model_path.")

    if (args.prompt is None) == (args.prompt_file is None):
        raise SystemExit("Provide exactly one of --prompt or --prompt_file")
    prompt = args.prompt
    if args.prompt_file:
        prompt = open(args.prompt_file, "r", encoding="utf-8").read()

    cfg = InferenceConfig(
        model_path=str(model_path),
        use_4bit=args.use_4bit,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
        seed=args.seed,
        deterministic=bool(args.deterministic),
    )
    obj = generate_json_only(prompt, cfg)
    print(json.dumps(obj, ensure_ascii=False))


if __name__ == "__main__":
    main()

