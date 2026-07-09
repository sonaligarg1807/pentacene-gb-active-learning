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
import subprocess
import shutil
import yaml
from datetime import datetime

CONFIG_PATH = "/home/sgarg/git-repo/pentacene-gb-active-learning/configs/mace_finetune_config.yaml"


def main():
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)

    os.makedirs(config["model_dir"], exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archived_config_path = os.path.join(
        config["model_dir"], f"config_used_{timestamp}.yaml"
    )
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
    ]

    print("Running command:")
    print(" ".join(command))
    print()

    result = subprocess.run(command)

    if result.returncode == 0:
        print(f"\nFine-tuning completed. Model saved to {config['model_dir']}")
    else:
        print(f"\nFine-tuning failed with exit code {result.returncode}.")


if __name__ == "__main__":
    main()