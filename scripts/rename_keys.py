"""
rename_keys.py

Renames 'energy' -> 'REF_energy' and 'forces' -> 'REF_forces' in the
labeled dataset, following MACE's recommended convention to avoid
ambiguity with ASE's internal calculator-derived keys.

Note: since ASE 3.23+, 'energy' and 'forces' read from extended XYZ files
are automatically attached to an internal SinglePointCalculator rather than
left in atoms.info/atoms.arrays. So we must retrieve them via
get_potential_energy()/get_forces() instead of dictionary lookups.

INPUT: data/labeled_data/train.xyz, valid.xyz
OUTPUT: overwrites the same files with renamed keys

HOW TO RUN:
    python scripts/rename_keys.py
"""

from ase.io import read, write


def rename_keys(filepath):
    atoms_list = read(filepath, index=":")

    for atoms in atoms_list:
        # Retrieve values via the calculator ASE auto-attached on read,
        # rather than trying to pop them from .info/.arrays directly.
        energy = atoms.get_potential_energy()
        forces = atoms.get_forces()

        # Clear the old calculator so it doesn't conflict when writing
        atoms.calc = None

        # Store under the new, MACE-recommended key names
        atoms.info["REF_energy"] = energy
        atoms.arrays["REF_forces"] = forces

    write(filepath, atoms_list)
    print(f"Renamed keys in {filepath} ({len(atoms_list)} structures)")


def main():
    rename_keys("/Users/sonaligarg/Desktop/Phd_data/git-repo/pentacene-gb-active-learning/data/labeled_data/train.xyz")
    rename_keys("/Users/sonaligarg/Desktop/Phd_data/git-repo/pentacene-gb-active-learning/data/labeled_data/valid.xyz")


if __name__ == "__main__":
    main()