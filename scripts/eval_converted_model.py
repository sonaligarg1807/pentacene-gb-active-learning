"""
eval_converted_model.py

Runs a MACE .model file on a validation .xyz file and reports
RMSE for energy and forces, so results can be compared against
the numbers MACE printed during training (Error-table).

HOW TO RUN:
    python scripts/eval_converted_model.py \
        <model.model> <valid.xyz> [device]
"""

import sys
import numpy as np
import torch
from ase.io import read

from mace.calculators import MACECalculator


def evaluate(model_path: str, xyz_path: str, device: str = "cpu"):
    calc = MACECalculator(model_paths=model_path, device=device)

    atoms_list = read(xyz_path, index=":")

    energy_errors = []
    force_errors = []

    for atoms in atoms_list:
        if "REF_energy" not in atoms.info or "REF_forces" not in atoms.arrays:
            print("Skipping a frame missing REF_energy/REF_forces")
            continue

        ref_energy = atoms.info["REF_energy"]
        ref_forces = atoms.arrays["REF_forces"]
        n_atoms = len(atoms)

        atoms.calc = calc
        pred_energy = atoms.get_potential_energy()
        pred_forces = atoms.get_forces()

        energy_errors.append((pred_energy - ref_energy) / n_atoms)
        force_errors.append((pred_forces - ref_forces).flatten())

    energy_errors = np.array(energy_errors)
    force_errors = np.concatenate(force_errors)

    rmse_energy_mev = np.sqrt(np.mean(energy_errors ** 2)) * 1000
    rmse_force_mev = np.sqrt(np.mean(force_errors ** 2)) * 1000

    ref_force_norm = np.sqrt(np.mean(force_errors ** 2))
    all_ref_forces = np.concatenate(
        [atoms.arrays["REF_forces"].flatten() for atoms in atoms_list
         if "REF_forces" in atoms.arrays]
    )
    relative_force_rmse = (
        np.sqrt(np.mean(force_errors ** 2)) /
        np.sqrt(np.mean(all_ref_forces ** 2)) * 100
    )

    print("\n===== Evaluation Results =====")
    print(f"Num configs evaluated: {len(energy_errors)}")
    print(f"RMSE Energy  : {rmse_energy_mev:.1f} meV/atom")
    print(f"RMSE Forces  : {rmse_force_mev:.1f} meV/A")
    print(f"Relative F RMSE: {relative_force_rmse:.2f} %")
    print("================================\n")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python eval_converted_model.py <model.model> <valid.xyz> [device]")
        sys.exit(1)

    model_path = sys.argv[1]
    xyz_path = sys.argv[2]
    device = sys.argv[3] if len(sys.argv) > 3 else "cpu"

    evaluate(model_path, xyz_path, device)