"""
labeler.py

Wraps the xtb-python ASE calculator to compute ground-truth energy and
forces for a given molecular structure using GFN2-xTB.

This is the "ground truth" source for your active learning loop — every
structure flagged as high-uncertainty later will be labeled using this
same function.
"""

from xtb.ase.calculator import XTB


def label_structure(atoms, method="GFN2-xTB", charge=0, multiplicity=1):
    """
    Runs a single-point GFN2-xTB calculation on an ASE Atoms object and
    returns its energy and forces.

    Parameters
    ----------
    atoms : ASE Atoms
        The molecular structure to label.
    method : str
        xTB method to use. "GFN2-xTB" is the standard, most broadly
        accurate tight-binding method.
    charge : int
        Total molecular charge (0 for neutral pentacene dimers).
    multiplicity : int
        Spin multiplicity (1 = singlet, closed-shell).

    Returns
    -------
    energy : float
        Total energy in eV (ASE's standard unit).
    forces : numpy.ndarray, shape (n_atoms, 3)
        Atomic forces in eV/Angstrom.
    """
    calc = XTB(method=method, charge=charge, uhf=multiplicity - 1)
    atoms.calc = calc

    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()

    return energy, forces