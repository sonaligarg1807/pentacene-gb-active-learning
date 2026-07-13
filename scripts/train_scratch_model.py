"""
train_scratch_model.py

Wrapper script that calls MACE's official training CLI (mace_run_train)
using the settings in configs/mace_scratch_config.yaml, and logs the run
for reproducibility.

Why a wrapper instead of a custom training loop:
MACE already provides a well-tested, official training command. Writing
a custom training loop would duplicate this functionality unnecessarily.
This wrapper just makes it easy to run consistently and keeps a record
of exactly which config was used for each run.

INPUT FILES NEEDED:
- data/labeled_data/train.xyz
- data/labeled_data/valid.xyz
- configs/mace_scratch_config.yaml

OUTPUT FILES PRODUCED:
- output/models/scratch_v1/  (trained model checkpoints, logs)

HOW TO RUN:
    python scripts/train_scratch_model.py
"""

import os
import subprocess
import shutil
import yaml
from datetime import datetime


CONFIG_PATH = "/home/sgarg/git-repo/pentacene-gb-active-learning/configs/mace_scratch_config.yaml"


def main():
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)

    os.makedirs(config["model_dir"], exist_ok=True)

    # Save a timestamped copy of the config used for this run, so every
    # training run is traceable later (important for reproducibility).
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archived_config_path = os.path.join(
        config["model_dir"], f"config_used_{timestamp}.yaml"
    )
    shutil.copy(CONFIG_PATH, archived_config_path)

    # Build the mace_run_train command from the config values.
    # mace_run_train reads most arguments directly as command-line flags.
    command = [
        "mace_run_train",
        f"--name={config['name']}",
        f"--seed={config['seed']}",
        f"--train_file={config['train_file']}",
        f"--valid_file={config['valid_file']}",
        f"--energy_key={config['energy_key']}",
        f"--E0s={config['e0s']}",
        f"--forces_key={config['forces_key']}",
        f"--model={config['model']}",
        f"--num_channels={config['num_channels']}",
        f"--max_L={config['max_L']}",
        f"--r_max={config['r_max']}",
        f"--num_interactions={config['num_interactions']}",
        f"--batch_size={config['batch_size']}",
        f"--max_num_epochs={config['max_num_epochs']}",
        f"--lr={config['lr']}",
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
        print(f"\nTraining completed successfully. Model saved to {config['model_dir']}")
    else:
        print(f"\nTraining failed with exit code {result.returncode}. "
              f"Check the log above for the error message.")


if __name__ == "__main__":
    main()