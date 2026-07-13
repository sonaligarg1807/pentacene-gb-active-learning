import sys
from pathlib import Path

import torch


def convert(checkpoint_path: str, template_model_path: str, output_path: str):
    checkpoint_path = Path(checkpoint_path)
    template_model_path = Path(template_model_path)
    output_path = Path(output_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not template_model_path.exists():
        raise FileNotFoundError(f"Template model not found: {template_model_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = torch.load(template_model_path, map_location="cpu", weights_only=False)

    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
    else:
        raise RuntimeError(f"Unexpected checkpoint type: {type(checkpoint)}")

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"WARNING: missing keys when loading: {missing}")
    if unexpected:
        print(f"WARNING: unexpected keys when loading: {unexpected}")

    model.eval()
    torch.save(model, output_path)
    print(f"Saved compiled model to {output_path}")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python checkpoint_to_model.py <checkpoint.pt> <template.model> <output.model>")
        sys.exit(1)

    convert(sys.argv[1], sys.argv[2], sys.argv[3])