import argparse
from pathlib import Path
from typing import Optional

import torch
from huggingface_hub import snapshot_download

PYTORCH_REPO = "facebook/sam3"


def load_from_hub(
    hf_repo: str = PYTORCH_REPO,
    local_dir: Optional[str] = None,
) -> Path:
    download_kwargs = {
        "repo_id": hf_repo,
        "allow_patterns": ["*.pt", "*.json"],
    }

    if local_dir:
        download_kwargs["local_dir"] = local_dir

    model_path = Path(snapshot_download(**download_kwargs))
    weights_file = model_path / "sam3.pt"

    if not weights_file.exists():
        raise FileNotFoundError(f"sam3.pt not found in {hf_repo}.")

    return weights_file


def load_weights(checkpoint_path: str):
    """Load PyTorch weights directly (no conversion needed)."""
    weights = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    return weights


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download SAM3 PyTorch weights")
    parser.add_argument(
        "--repo",
        default=PYTORCH_REPO,
        type=str,
        help=f"HuggingFace repo (default: {PYTORCH_REPO})",
    )
    parser.add_argument(
        "--local-dir",
        type=str,
        default=None,
        help="Local path to cache weights.",
    )
    args = parser.parse_args()

    print(f"Downloading PyTorch weights from {args.repo}...")
    weights_path = load_from_hub(args.repo, args.local_dir)
    print(f"Weights available at: {weights_path}")
