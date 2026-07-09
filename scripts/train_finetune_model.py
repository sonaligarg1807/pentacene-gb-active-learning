"""
train_finetune_model.py

Fine-tunes a pretrained MACE foundation model (MACE-OFF23) on the
pentacene GB seed dataset, using mace_run_train's built-in
--foundation_model argument.

INPUT FILES NEEDED:
- data/labeled_data/train.xyz, valid.xyz
- configs/mace_finetune_config.yaml
- Pretrained foundation model checkpoint (downloaded separately, see Step 1)

OUTPUT FILES PRODUCED:
- output/models/finetune_v1/ (fine-tuned model checkpoints, logs)

HOW TO RUN (typically inside a SLURM job on JUSTUS2):
    python scripts/train_finetune_model.py
"""

import os
import shutil
import subprocess
import yaml
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "mace_finetune_config.yaml"


def main():
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Config file not found: {CONFIG_PATH}")

    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)

    required_keys = [
        "name",
        "seed",
        "train_file",
        "valid_file",
        "energy_key",
        "forces_key",
        "foundation_model",
        "lr",
        "batch_size",
        "max_num_epochs",
        "weight_decay",
        "energy_weight",
        "forces_weight",
        "model_dir",
        "log_dir",
        "checkpoints_dir",
        "device",
    ]
    missing = [k for k in required_keys if k not in config]
    if missing:
        raise KeyError(f"Missing required config keys: {missing}")

    for key in ["model_dir", "log_dir", "checkpoints_dir"]:
        os.makedirs(config[key], exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archived_config_path = Path(config["model_dir"]) / f"config_used_{timestamp}.yaml"
    shutil.copy(CONFIG_PATH, archived_config_path)

    command = [
        "mace_run_train",
        f"--name={config['name']}",
        f"--seed={config['seed']}",
        f"--train_file={config['train_file']}",
        f"--valid_file={config['valid_file']}",
        f"--energy_key={config['energy_key']}",
        f"--forces_key={config['forces_key']}",
        f"--foundation_model={config['foundation_model']}",
        f"--lr={config['lr']}",
        f"--batch_size={config['batch_size']}",
        f"--max_num_epochs={config['max_num_epochs']}",
        f"--weight_decay={config['weight_decay']}",
        f"--energy_weight={config['energy_weight']}",
        f"--forces_weight={config['forces_weight']}",
        f"--model_dir={config['model_dir']}",
        f"--log_dir={config['log_dir']}",
        f"--checkpoints_dir={config['checkpoints_dir']}",
        f"--device={config['device']}",
        f"--multiheads_finetuning={config['multiheads_finetuning']}",
    ]

    optional_keys = {
        "default_dtype": "--default_dtype",
        "E0s": "--E0s",
        "scaling": "--scaling",
        "ema": "--ema",
        "ema_decay": "--ema_decay",
        "amsgrad": "--amsgrad",
    }

    for key, flag in optional_keys.items():
        if key in config:
            value = config[key]
            if isinstance(value, bool):
                if value:
                    command.append(flag)
            else:
                command.append(f"{flag}={value}")

    print("Running command:")
    print(" ".join(command))
    print()

    subprocess.run(command, check=True)

    print(f"\nFine-tuning completed. Model saved to {config['model_dir']}")


if __name__ == "__main__":
    main()