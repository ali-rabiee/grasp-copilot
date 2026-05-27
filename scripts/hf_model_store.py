import argparse
from pathlib import Path

import _bootstrap  # noqa: F401


REQUIRED_FILES = (
    "config.json",
    "generation_config.json",
    "model.safetensors.index.json",
    "tokenizer_config.json",
)


def _is_complete_model_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    if not all((path / name).is_file() for name in REQUIRED_FILES):
        return False
    return any(path.glob("model*.safetensors"))


def _iter_complete_model_dirs(root: Path) -> list[Path]:
    return sorted(p for p in root.iterdir() if _is_complete_model_dir(p))


def _repo_id(namespace: str, model_dir: Path, prefix: str) -> str:
    repo_name = f"{prefix}{model_dir.name}" if prefix else model_dir.name
    return f"{namespace.rstrip('/')}/{repo_name}"


def cmd_list(args: argparse.Namespace) -> None:
    root = Path(args.source_root).expanduser()
    complete = _iter_complete_model_dirs(root)
    if not complete:
        print(f"No complete merged model directories found under {root}")
        return
    for model_dir in complete:
        print(model_dir)


def cmd_upload(args: argparse.Namespace) -> None:
    from huggingface_hub import HfApi

    source_root = Path(args.source_root).expanduser()
    if args.model:
        model_dirs = [source_root / args.model]
    else:
        model_dirs = _iter_complete_model_dirs(source_root)

    if not model_dirs:
        raise SystemExit(f"No complete merged model directories found under {source_root}")

    api = HfApi()
    for model_dir in model_dirs:
        if not _is_complete_model_dir(model_dir):
            raise SystemExit(f"Not a complete merged model directory: {model_dir}")
        repo_id = args.repo_id or _repo_id(args.namespace, model_dir, args.prefix)
        print(f"[hf] uploading {model_dir} -> {repo_id}")
        api.create_repo(repo_id=repo_id, repo_type="model", private=not args.public, exist_ok=True)
        api.upload_folder(
            repo_id=repo_id,
            repo_type="model",
            folder_path=str(model_dir),
            commit_message=args.commit_message,
        )
        print(f"[hf] uploaded {repo_id}")


def cmd_download(args: argparse.Namespace) -> None:
    from huggingface_hub import snapshot_download

    target_dir = Path(args.target_dir).expanduser()
    target_dir.mkdir(parents=True, exist_ok=True)
    local_dir = target_dir / args.local_name if args.local_name else target_dir / args.repo_id.split("/")[-1]
    print(f"[hf] downloading {args.repo_id} -> {local_dir}")
    snapshot_download(
        repo_id=args.repo_id,
        repo_type="model",
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
    )
    print(f"[hf] downloaded {args.repo_id} to {local_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload/download merged grasp-copilot models on Hugging Face Hub.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    list_p = sub.add_parser("list", help="List complete merged model directories under a source root.")
    list_p.add_argument("--source-root", default="/media/ali/USB/old")
    list_p.set_defaults(func=cmd_list)

    upload_p = sub.add_parser("upload", help="Upload one complete model directory, or all complete models under source root.")
    upload_p.add_argument("--source-root", default="/media/ali/USB/old")
    upload_p.add_argument("--namespace", help="Your Hugging Face username or org, e.g. ali-rabiee.")
    upload_p.add_argument("--prefix", default="grasp-copilot-", help="Prefix added to repo names when --repo-id is not used.")
    upload_p.add_argument("--model", help="Directory name under --source-root to upload. Omit to upload all complete models.")
    upload_p.add_argument("--repo-id", help="Exact repo id for single-model upload, e.g. user/grasp-copilot-oracle-woz.")
    upload_p.add_argument("--public", action="store_true", help="Create public repos. Default is private.")
    upload_p.add_argument("--commit-message", default="Upload merged grasp-copilot model")
    upload_p.set_defaults(func=cmd_upload)

    download_p = sub.add_parser("download", help="Download a model repo into a local models directory.")
    download_p.add_argument("--repo-id", required=True, help="Hub repo id, e.g. user/grasp-copilot-qwen2_5_3b_oracle_woz_lora")
    download_p.add_argument("--target-dir", default="models")
    download_p.add_argument("--local-name", help="Local directory name under --target-dir. Defaults to repo name.")
    download_p.set_defaults(func=cmd_download)

    args = parser.parse_args()
    if args.cmd == "upload" and not args.repo_id and not args.namespace:
        raise SystemExit("upload requires --namespace unless --repo-id is provided")
    if args.cmd == "upload" and args.repo_id and not args.model:
        raise SystemExit("--repo-id can only be used with --model, so the destination is unambiguous")
    args.func(args)


if __name__ == "__main__":
    main()
