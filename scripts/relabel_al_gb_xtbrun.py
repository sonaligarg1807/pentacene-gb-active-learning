"""
label_gb_dimers.py

Runs GFN2-xTB single-point calculations on the top-uncertainty GB dimer
structures selected via committee active learning (Step 6e).

INPUT FILES NEEDED:
- data/gb_dimers/selected_for_xtb/all_top20pct_combined.xyz
- configs/gb_labeling_config.yaml

OUTPUT FILES PRODUCED:
- data/labeled_data/gb_al_round1_labeled.xyz
- data/labeled_data/gb_al_round1_labeling_failures.csv

HOW TO RUN:
    python scripts/label_gb_dimers.py
"""

import os
import sys
import csv
import yaml
from ase.io import read, write
from ase.calculators.singlepoint import SinglePointCalculator

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.labeler import label_structure


def load_config(config_path="/home/sgarg/git-repo/pentacene-gb-active-learning/configs/relabel_al_gb_config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    config = load_config()

    output_dir = os.path.dirname(config["output_labeled_file"])
    os.makedirs(output_dir, exist_ok=True)

    # CHANGED: read all frames from one multi-frame .xyz instead of globbing a directory
    atoms_list = read(config["input_xyz"], index=":")
    print(f"Found {len(atoms_list)} structures to label.")

    labeled_atoms_list = []
    failures = []

    for i, atoms in enumerate(atoms_list):
        # CHANGED: use frame index + comment info instead of a filename
        identifier = atoms.info.get("comment", f"frame_{i}")

        try:
            energy, forces = label_structure(
                atoms,
                method=config["xtb_method"],
                charge=config["charge"],
                multiplicity=config["multiplicity"],
            )

            atoms.calc = SinglePointCalculator(atoms, energy=energy, forces=forces)
            labeled_atoms_list.append(atoms)

            if (i + 1) % 20 == 0:
                print(f"[{i + 1}/{len(atoms_list)}] Labeled frame {i} "
                      f"(energy={energy:.4f} eV)")

        except Exception as e:
            print(f"FAILED on frame {i} ({identifier}): {e}")
            failures.append({"frame_index": i, "identifier": identifier, "error": str(e)})

    if labeled_atoms_list:
        write(config["output_labeled_file"], labeled_atoms_list)
        print(f"\nSaved {len(labeled_atoms_list)} labeled structures to "
              f"{config['output_labeled_file']}")
    else:
        print("\nNo structures were successfully labeled. Check errors above.")

    if failures:
        with open(config["failed_log_file"], "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["frame_index", "identifier", "error"])
            writer.writeheader()
            writer.writerows(failures)
        print(f"{len(failures)} structures failed. See {config['failed_log_file']}")
    else:
        print("No failures.")


if __name__ == "__main__":
    main()