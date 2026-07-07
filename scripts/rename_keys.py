"""
rename_keys.py

Renames 'energy' -> 'REF_energy' and 'forces' -> 'REF_forces' in the
labeled dataset, following MACE's recommended convention to avoid
ambiguity with ASE's internal calculator-derived keys.

INPUT: data/labeled_data/train.xyz, valid.xyz
OUTPUT: overwrites the same files with renamed keys

HOW TO RUN:
    python scripts/rename_keys.py
"""

from ase.io import read, write


def rename_keys(filepath):
    atoms_list = read(filepath, index=":")
    for atoms in atoms_list:
        energy = atoms.info.pop("energy")
        forces = atoms.arrays.pop("forces")
        atoms.info["REF_energy"] = energy
        atoms.arrays["REF_forces"] = forces
    write(filepath, atoms_list)
    print(f"Renamed keys in {filepath} ({len(atoms_list)} structures)")


def main():
    rename_keys("data/labeled_data/train.xyz")
    rename_keys("data/labeled_data/valid.xyz")


if __name__ == "__main__":
    main()