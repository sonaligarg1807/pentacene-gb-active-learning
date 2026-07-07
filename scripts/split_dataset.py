"""
split_dataset.py

Splits the labeled dataset into training and validation sets, which MACE
requires as separate files during training.

INPUT FILES NEEDED:
- data/labeled_data/seed_dataset_labeled.xyz

OUTPUT FILES PRODUCED:
- data/labeled_data/train.xyz
- data/labeled_data/valid.xyz

HOW TO RUN:
    python scripts/split_dataset.py
"""

import random
from ase.io import read, write

INPUT_FILE = "/Users/sonaligarg/Desktop/Phd_data/git-repo/pentacene-gb-active-learning/data/labeled_data/seed_dataset_labeled.xyz"
TRAIN_FILE = "/Users/sonaligarg/Desktop/Phd_data/git-repo/pentacene-gb-active-learning/data/labeled_data/train.xyz"
VALID_FILE = "/Users/sonaligarg/Desktop/Phd_data/git-repo/pentacene-gb-active-learning/data/labeled_data/valid.xyz"

VALID_FRACTION = 0.15   # 15% held out for validation
RANDOM_SEED = 42


def main():
    all_structures = read(INPUT_FILE, index=":")
    print(f"Loaded {len(all_structures)} structures.")

    random.seed(RANDOM_SEED)
    indices = list(range(len(all_structures)))
    random.shuffle(indices)

    n_valid = max(1, int(len(all_structures) * VALID_FRACTION))
    valid_indices = set(indices[:n_valid])

    train_structures = [s for i, s in enumerate(all_structures) if i not in valid_indices]
    valid_structures = [s for i, s in enumerate(all_structures) if i in valid_indices]

    write(TRAIN_FILE, train_structures)
    write(VALID_FILE, valid_structures)

    print(f"Train set: {len(train_structures)} structures -> {TRAIN_FILE}")
    print(f"Valid set: {len(valid_structures)} structures -> {VALID_FILE}")


if __name__ == "__main__":
    main()