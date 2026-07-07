"""
compute_atomic_energies.py

Computes isolated-atom reference energies (E0s) for Carbon and Hydrogen
using GFN2-xTB. These values are required by MACE to properly reference
its energy predictions.

INPUT FILES NEEDED: none (builds single isolated atoms directly)

OUTPUT: prints E0 values to paste into your MACE config

HOW TO RUN:
    python scripts/compute_atomic_energies.py
"""

from ase import Atoms
from xtb.ase.calculator import XTB


def compute_e0(symbol):
    # Isolated atom in a large empty box, no periodic boundary conditions
    atom = Atoms(symbol, positions=[[0, 0, 0]])
    atom.calc = XTB(method="GFN2-xTB")
    energy = atom.get_potential_energy()
    return energy


def main():
    for symbol in ["H", "C"]:
        e0 = compute_e0(symbol)
        print(f"{symbol}: E0 = {e0:.6f} eV")


if __name__ == "__main__":
    main()