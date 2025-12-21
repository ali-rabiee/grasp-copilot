from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from . import data as data_lib
from .utils import ensure_dir, save_run_config, set_seed

if TYPE_CHECKING:
    # Optional dependency; only needed when training with LoRA.
    from peft import LoraConfig  # type: ignore[import]
    from peft import PeftModel  # type: ignore[import]
    from peft import get_peft_model, prepare_model_for_kbit_training  # type: ignore[import]


DEFAULT_MODEL = "Qwen/Qwen3-0.6B"


@dataclass(frozen=True, slots=True)
class TrainArgs:
    model_name: str = DEFAULT_MODEL
    train_path: str = ""
    valid_path: Optional[str] = None
    output_dir: str = "models/lora"
    # NOTE: 2048 can OOM on ~16GB GPUs depending on model + eval settings.
    # 1024 is a safer default; you can still pass --max_seq_length 2048 explicitly.
    max_seq_length: int = 1024
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    lr: float = 2e-4
    num_train_epochs: float = 1.0
    # If > 0, overrides num_train_epochs and trains for exactly this many optimizer steps.
    max_steps: int = -1
    # Slightly smaller LoRA reduces VRAM/CPU overhead while keeping decent quality.
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    # 4-bit is the most reliable way to fit 7B models on ~16GB GPUs.
    use_4bit: bool = True
    seed: int = 0
    eval_steps: int = 200
    eval_accumulation_steps: int = 1
    prediction_loss_only: bool = True
    disable_eval: bool = False
    # Avoid large checkpoint artifacts (optimizer/scheduler state) by default.
    # We'll still save the final adapter via model.save_pretrained() at the end.
    save_strategy: str = "no"  # "no" | "steps" | "epoch"
    save_steps: int = 200
    save_only_model: bool = True
    save_total_limit: int = 2
    logging_steps: int = 20
    warmup_ratio: float = 0.03
    report_to: str = "none"


def _make_peft_config(args: TrainArgs):
    from peft import LoraConfig  # type: ignore[import]

    # Reasonable default for Qwen-family decoder blocks.
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    return LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )


def _load_model_and_tokenizer(args: TrainArgs):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True, fix_mistral_regex=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # Prefer bf16 when supported; otherwise fall back to fp16 on CUDA.
    use_bf16 = bool(torch.cuda.is_available() and getattr(torch.cuda, "is_bf16_supported", lambda: False)())

    quant_cfg = None
    if args.use_4bit:
        try:
            from transformers import BitsAndBytesConfig

            quant_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                # IMPORTANT: If we use bf16 compute, we must NOT enable fp16 GradScaler in Trainer.
                bnb_4bit_compute_dtype=torch.bfloat16 if use_bf16 else (torch.float16 if torch.cuda.is_available() else torch.float16),
            )
        except Exception as e:
            raise RuntimeError("use_4bit requires bitsandbytes + BitsAndBytesConfig") from e

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        dtype="auto",
        quantization_config=quant_cfg,
        device_map="auto" if args.use_4bit else None,
    )
    return model, tok


def _chat_to_text(tok, messages: List[Dict[str, str]]) -> str:
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)


def train_sft_lora(args: TrainArgs) -> None:
    set_seed(args.seed)
    ensure_dir(args.output_dir)

    # Validate contract early for fail-fast UX.
    data_lib.validate_dataset_contract_jsonl(args.train_path)
    if args.valid_path:
        data_lib.validate_dataset_contract_jsonl(args.valid_path)

    from datasets import load_dataset
    from peft import get_peft_model, prepare_model_for_kbit_training  # type: ignore[import]
    from transformers import TrainingArguments

    import torch
    use_bf16 = bool(torch.cuda.is_available() and getattr(torch.cuda, "is_bf16_supported", lambda: False)())

    model, tok = _load_model_and_tokenizer(args)

    if args.use_4bit:
        model = prepare_model_for_kbit_training(model)

    model.gradient_checkpointing_enable()

    peft_cfg = _make_peft_config(args)
    model = get_peft_model(model, peft_cfg)

    train_ds = load_dataset("json", data_files=args.train_path, split="train")
    eval_ds = None
    if args.valid_path and not args.disable_eval:
        eval_ds = load_dataset("json", data_files=args.valid_path, split="train")

    def to_messages(batch):
        msgs = []
        for i in range(len(batch["id"])):
            ex = data_lib.DatasetExample(
                id=batch["id"][i],
                instruction=batch["instruction"][i],
                input=batch["input"][i],
                output=batch["output"][i],
            )
            msgs.append(data_lib.dataset_contract_to_qwen_chat_messages(ex)["messages"])
        return {"messages": msgs}

    # Datasets types can be loose; cast column names to list for type-checkers.
    from typing import cast

    train_cols = cast(List[str], list(train_ds.column_names or []))
    train_ds = train_ds.map(to_messages, batched=True, remove_columns=train_cols)
    if eval_ds is not None:
        eval_cols = cast(List[str], list(eval_ds.column_names or []))
        eval_ds = eval_ds.map(to_messages, batched=True, remove_columns=eval_cols)

    def formatting_func(examples):
        """
        TRL calls formatting_func either:
          - batched: examples["messages"] is List[List[Dict[str,str]]]
          - unbatched: examples["messages"] is List[Dict[str,str]] (single conversation)
        """
        msgs = examples["messages"]
        # Unbatched: one conversation = list of role/content dicts.
        if isinstance(msgs, list) and (len(msgs) == 0 or isinstance(msgs[0], dict)):
            return _chat_to_text(tok, msgs)
        # Batched: list of conversations.
        return [_chat_to_text(tok, m) for m in msgs]

    try:
        from trl import SFTTrainer  # type: ignore[attr-defined]
    except Exception:
        try:
            from trl.trainer.sft_trainer import SFTTrainer  # type: ignore
        except Exception as e:
            raise RuntimeError("trl is required for SFT training") from e

    # Transformers renamed `evaluation_strategy` -> `eval_strategy` in newer versions.
    # Also, different versions accept different TrainingArguments kwargs. We'll filter by
    # the installed signature to keep compatibility.
    eval_strategy = "steps" if eval_ds is not None else "no"
    targs_kwargs: Dict[str, Any] = dict(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.lr,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        logging_steps=args.logging_steps,
        warmup_ratio=args.warmup_ratio,
        report_to=args.report_to,
        # IMPORTANT: don't run fp16 GradScaler on bf16 tensors (can crash during unscale/clip).
        fp16=bool(torch.cuda.is_available() and not use_bf16),
        bf16=bool(torch.cuda.is_available() and use_bf16),
        remove_unused_columns=False,
        gradient_checkpointing=True,
        # Reduce eval memory pressure: don't gather/store full logits unless needed.
        prediction_loss_only=args.prediction_loss_only,
        # Reduce eval host/GPU memory when eval sets are large.
        eval_accumulation_steps=args.eval_accumulation_steps,
    )
    # Checkpointing: default to "no" to avoid huge optimizer checkpoints; always save final adapter ourselves.
    if args.save_strategy != "no":
        targs_kwargs["save_strategy"] = args.save_strategy
        targs_kwargs["save_steps"] = args.save_steps
        targs_kwargs["save_total_limit"] = args.save_total_limit
        targs_kwargs["save_only_model"] = args.save_only_model
    if eval_ds is not None:
        targs_kwargs["eval_steps"] = args.eval_steps

    import inspect

    ta_sig = inspect.signature(TrainingArguments.__init__)
    filtered_targs_kwargs = {k: v for k, v in targs_kwargs.items() if k in ta_sig.parameters}
    try:
        if "eval_strategy" in ta_sig.parameters:
            targs = TrainingArguments(eval_strategy=eval_strategy, **filtered_targs_kwargs)  # type: ignore[arg-type]
        else:
            targs = TrainingArguments(evaluation_strategy=eval_strategy, **filtered_targs_kwargs)  # type: ignore[arg-type]
    except TypeError as e:
        # Fallback: in case of unexpected signature differences.
        targs = TrainingArguments(output_dir=args.output_dir)  # type: ignore[arg-type]
        raise RuntimeError(
            "Could not construct TrainingArguments with the current transformers version. "
            "Please report your transformers version and the CLI args you used."
        ) from e

    # TRL's SFTTrainer API has changed across versions (tokenizer -> processing_class,
    # max_seq_length -> max_length, packing added/removed, etc). Build kwargs based on
    # the installed signature to keep compatibility across environments.
    import inspect

    sig = inspect.signature(SFTTrainer.__init__)

    # Always-required-ish fields for our usage.
    candidate_kwargs: Dict[str, Any] = {
        "model": model,
        "train_dataset": train_ds,
        "eval_dataset": eval_ds,
        "formatting_func": formatting_func,
        "args": targs,
    }

    # Tokenizer / processing class naming differs by TRL version.
    if "tokenizer" in sig.parameters:
        candidate_kwargs["tokenizer"] = tok
    elif "processing_class" in sig.parameters:
        candidate_kwargs["processing_class"] = tok
    else:
        raise RuntimeError("Unsupported trl.SFTTrainer API (no tokenizer/processing_class parameter found)")

    # Sequence length naming differs by TRL version.
    if "max_seq_length" in sig.parameters:
        candidate_kwargs["max_seq_length"] = args.max_seq_length
    elif "max_length" in sig.parameters:
        candidate_kwargs["max_length"] = args.max_seq_length

    # Packing is optional and not supported in all versions.
    if "packing" in sig.parameters:
        candidate_kwargs["packing"] = True

    # Filter only kwargs supported by the installed TRL signature.
    sft_kwargs = {k: v for k, v in candidate_kwargs.items() if k in sig.parameters}

    trainer = SFTTrainer(**sft_kwargs)  # type: ignore[call-arg]
    trainer.train()

    model.save_pretrained(args.output_dir)
    tok.save_pretrained(args.output_dir)
    save_run_config(os.path.join(args.output_dir, "run_config.json"), args)


def merge_lora(base_model_name: str, adapter_dir: str, output_dir: str) -> None:
    ensure_dir(output_dir)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel  # type: ignore[import]

    tok = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True, fix_mistral_regex=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(base_model_name, trust_remote_code=True, dtype="auto")
    model = PeftModel.from_pretrained(model, adapter_dir)
    merged = model.merge_and_unload()
    merged.save_pretrained(output_dir, safe_serialization=True)
    tok.save_pretrained(output_dir)


def smoke_train_step() -> None:
    """
    CPU-only smoke check: tiny random LM forward/backward + optimizer step.
    Avoids any HF downloads (network-restricted environments).
    """
    import torch
    from transformers import GPT2Config, GPT2LMHeadModel

    cfg = GPT2Config(
        vocab_size=128,
        n_positions=64,
        n_embd=32,
        n_layer=2,
        n_head=2,
    )
    model = GPT2LMHeadModel(cfg)
    model.train()

    # Fake token batch.
    input_ids = torch.randint(0, cfg.vocab_size, (2, 32))
    labels = input_ids.clone()

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    out = model(input_ids=input_ids, labels=labels)
    loss = out.loss
    loss.backward()
    opt.step()
    opt.zero_grad(set_to_none=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", type=str, default=DEFAULT_MODEL)
    ap.add_argument("--train_path", type=str, required=True)
    ap.add_argument("--valid_path", type=str, default=None)
    ap.add_argument("--output_dir", type=str, required=True)
    ap.add_argument("--max_seq_length", type=int, default=1024)
    ap.add_argument("--per_device_train_batch_size", type=int, default=1)
    ap.add_argument("--per_device_eval_batch_size", type=int, default=1)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--num_train_epochs", type=float, default=1.0)
    ap.add_argument("--max_steps", type=int, default=-1, help="If > 0, train for exactly this many optimizer steps (overrides epochs).")
    ap.add_argument("--lora_r", type=int, default=8)
    ap.add_argument("--lora_alpha", type=int, default=16)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--use_4bit", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval_steps", type=int, default=200)
    ap.add_argument("--eval_accumulation_steps", type=int, default=1)
    ap.add_argument("--prediction_loss_only", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--disable_eval", action="store_true", help="If set, ignore --valid_path and do not run evaluation during training.")
    ap.add_argument("--save_strategy", type=str, default="no", choices=["no", "steps", "epoch"])
    ap.add_argument("--save_steps", type=int, default=500)
    ap.add_argument("--save_only_model", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--save_total_limit", type=int, default=2)
    ap.add_argument("--logging_steps", type=int, default=20)
    ap.add_argument("--warmup_ratio", type=float, default=0.03)
    ap.add_argument("--report_to", type=str, default="none")
    args = ap.parse_args()

    train_sft_lora(
        TrainArgs(
            model_name=args.model_name,
            train_path=args.train_path,
            valid_path=args.valid_path,
            output_dir=args.output_dir,
            max_seq_length=args.max_seq_length,
            per_device_train_batch_size=args.per_device_train_batch_size,
            per_device_eval_batch_size=args.per_device_eval_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            lr=args.lr,
            num_train_epochs=args.num_train_epochs,
            max_steps=args.max_steps,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            use_4bit=args.use_4bit,
            seed=args.seed,
            eval_steps=args.eval_steps,
            eval_accumulation_steps=args.eval_accumulation_steps,
            prediction_loss_only=args.prediction_loss_only,
            disable_eval=args.disable_eval,
            save_strategy=args.save_strategy,
            save_steps=args.save_steps,
            save_only_model=args.save_only_model,
            save_total_limit=args.save_total_limit,
            logging_steps=args.logging_steps,
            warmup_ratio=args.warmup_ratio,
            report_to=args.report_to,
        )
    )


if __name__ == "__main__":
    main()

