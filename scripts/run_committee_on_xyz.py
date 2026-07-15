"""
run_committee_on_xyz.py

Uses MACECommittee to evaluate a set of structures in an .xyz file
and writes out mean/std energy and forces per structure.

HOW TO RUN:
    python scripts/run_committee_on_xyz.py path/to/input.xyz path/to/output.json
"""

import sys
import json
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))

from ase.io import read
from committee.committee import MACECommittee  # adjust import if needed


def load_structures(xyz_path):
    # ASE can read multiple frames from one .xyz
    atoms_list = read(xyz_path, index=":")
    return atoms_list


def main():
    if len(sys.argv) != 3:
        print("Usage: python run_committee_on_xyz.py input.xyz output.json")
        sys.exit(1)

    input_xyz = Path(sys.argv[1])
    output_json = Path(sys.argv[2])

    if not input_xyz.exists():
        raise FileNotFoundError(f"Input xyz not found: {input_xyz}")

    # 1. Load structures
    atoms_list = load_structures(input_xyz)

    # 2. Load committee
    committee = MACECommittee()

    # 3. Evaluate batch
    results = []
    for i, atoms in enumerate(atoms_list):
        res = committee.predict_single(atoms)

        mean_forces = res["mean_forces"]  # shape: (n_atoms, 3)
        std_forces = res["std_forces"]    # shape: (n_atoms, 3)

        mean_force = float(np.mean(np.abs(mean_forces)))
        mean_std_force = float(np.mean(std_forces))

        results.append({
            "index": i,
            "mean_energy": float(res["mean_energy"]),
            "std_energy": float(res["std_energy"]),
            "mean_force": mean_force,
            "mean_std_force": mean_std_force,
        })
    # 4. Save results
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Wrote committee results for {len(results)} structures to {output_json}")


if __name__ == "__main__":
    main()