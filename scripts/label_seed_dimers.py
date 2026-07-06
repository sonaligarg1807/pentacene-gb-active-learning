"""
label_seed_dimers.py

Runs GFN2-xTB single-point calculations on all generated seed dimer
structures, computing ground-truth energy and forces for each one.

Results are saved into a single extended XYZ file (the standard format
MACE and other MLIP frameworks expect for training), with energy and
forces embedded as per-structure and per-atom properties.

INPUT FILES NEEDED:
- data/seed_structures/generated_dimers/*.xyz  (304 structures from Step 2)
- configs/labeling_config.yaml

OUTPUT FILES PRODUCED:
- data/labeled_data/seed_dataset_labeled.xyz   (all labeled structures, MACE-ready)
- data/labeled_data/labeling_failures.csv      (log of any structures that failed)

HOW TO RUN:
    python scripts/label_seed_dimers.py

NOTE: This will take some time (roughly a few seconds per structure with
GFN2-xTB), so for 304 structures expect this to run for several minutes.
"""

import os
import sys
import csv
import glob
import yaml
from ase.io import read, write
from ase.calculators.singlepoint import SinglePointCalculator

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.labeler import label_structure


def load_config(config_path="configs/labeling_config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    config = load_config()

    output_dir = os.path.dirname(config["output_labeled_file"])
    os.makedirs(output_dir, exist_ok=True)

    input_files = sorted(glob.glob(os.path.join(config["input_dir"], "*.xyz")))
    print(f"Found {len(input_files)} structures to label.")

    labeled_atoms_list = []
    failures = []

    for i, filepath in enumerate(input_files):
        filename = os.path.basename(filepath)

        try:
            atoms = read(filepath)

            energy, forces = label_structure(
                atoms,
                method=config["xtb_method"],
                charge=config["charge"],
                multiplicity=config["multiplicity"],
            )

            # Attach results as a SinglePointCalculator so they get written
            # correctly into the extended XYZ file (energy + forces per atom)
            atoms.calc = SinglePointCalculator(atoms, energy=energy, forces=forces)
            labeled_atoms_list.append(atoms)

            if (i + 1) % 20 == 0:
                print(f"[{i + 1}/{len(input_files)}] Labeled {filename} "
                      f"(energy={energy:.4f} eV)")

        except Exception as e:
            print(f"FAILED on {filename}: {e}")
            failures.append({"filename": filename, "error": str(e)})

    # ---- Save all labeled structures into one extended XYZ file ----
    if labeled_atoms_list:
        write(config["output_labeled_file"], labeled_atoms_list)
        print(f"\nSaved {len(labeled_atoms_list)} labeled structures to "
              f"{config['output_labeled_file']}")
    else:
        print("\nNo structures were successfully labeled. Check errors above.")

    # ---- Save failure log ----
    if failures:
        with open(config["failed_log_file"], "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["filename", "error"])
            writer.writeheader()
            writer.writerows(failures)
        print(f"{len(failures)} structures failed. See {config['failed_log_file']}")
    else:
        print("No failures.")


if __name__ == "__main__":
    main()

