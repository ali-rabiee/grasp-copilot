from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    # Optional dependency; imported at runtime only when adapter_path is provided.
    from peft import PeftModel  # type: ignore[import]  # pragma: no cover

from .utils import json_loads_strict, set_seed


@dataclass(frozen=True, slots=True)
class InferenceConfig:
    model_name: str
    adapter_path: Optional[str]
    merged_model_path: Optional[str]
    temperature: float = 0.2
    top_p: float = 0.9
    max_new_tokens: int = 512
    seed: int = 0


def _build_messages(prompt: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": "You are a helpful assistant. Output ONLY valid JSON."},
        {"role": "user", "content": prompt},
    ]


def _load_model_and_tokenizer(cfg: InferenceConfig):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_path = cfg.merged_model_path or cfg.model_name
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, fix_mistral_regex=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, dtype="auto")

    if cfg.adapter_path:
        from peft import PeftModel  # type: ignore[import]

        model = PeftModel.from_pretrained(model, cfg.adapter_path)

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
    ap.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--adapter_path", type=str, default=None)
    ap.add_argument("--merged_model_path", type=str, default=None)
    ap.add_argument("--prompt", type=str, default=None)
    ap.add_argument("--prompt_file", type=str, default=None)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if (args.prompt is None) == (args.prompt_file is None):
        raise SystemExit("Provide exactly one of --prompt or --prompt_file")
    prompt = args.prompt
    if args.prompt_file:
        prompt = open(args.prompt_file, "r", encoding="utf-8").read()

    cfg = InferenceConfig(
        model_name=args.model_name,
        adapter_path=args.adapter_path,
        merged_model_path=args.merged_model_path,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
        seed=args.seed,
    )
    obj = generate_json_only(prompt, cfg)
    print(json.dumps(obj, ensure_ascii=False))


if __name__ == "__main__":
    main()

