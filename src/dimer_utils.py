"""
dimer_utils.py

Helper functions for building pentacene dimer geometries using the SAME
misorientation convention as the bicrystal grain-boundary structures:
a rotation axis (a-axis, b-axis, or a plane normal) and a rotation angle
(0-90 degrees), with an automatic overlap check to prevent molecules
from clashing after rotation.
"""

import numpy as np
from ase.io import read


def load_monomer(xyz_path):
    """
    Reads a clean, single-molecule .xyz file (e.g. pentacene_monomer.xyz)
    and returns it as an ASE Atoms object.
    """
    molecule = read(xyz_path)
    return molecule


def center_molecule(molecule):
    """
    Shifts a molecule so its geometric center is at the origin (0, 0, 0).
    This makes rotations and translations predictable and easy to reason
    about.
    """
    positions = molecule.get_positions()
    center = positions.mean(axis=0)
    molecule.translate(-center)
    return molecule


def rotate_molecule_around_axis(molecule, angle_degrees, axis_vector):
    """
    Rotates a molecule by a given angle around an arbitrary axis vector.

    Parameters
    ----------
    molecule : ASE Atoms
        The molecule to rotate.
    angle_degrees : float
        Rotation angle in degrees (matches your bicrystal misorientation
        angle, e.g. 0, 15, 30, ... 90).
    axis_vector : array-like of length 3
        The rotation axis, e.g. the crystallographic a-axis, b-axis, or a
        plane normal. Does not need to be pre-normalized; ASE handles that.

    Returns
    -------
    rotated : ASE Atoms
        A new Atoms object, rotated around its own center of mass.
    """
    rotated = molecule.copy()
    rotated.rotate(angle_degrees, axis_vector, center="COM")
    return rotated


def min_interatomic_distance(monomer_a, monomer_b):
    """
    Computes the minimum distance between any atom in monomer_a and any
    atom in monomer_b. Used to detect whether two molecules are
    overlapping / clashing.
    """
    positions_a = monomer_a.get_positions()
    positions_b = monomer_b.get_positions()

    # Compute all pairwise distances between the two atom sets
    diff = positions_a[:, np.newaxis, :] - positions_b[np.newaxis, :, :]
    distances = np.linalg.norm(diff, axis=2)

    return distances.min()


def build_dimer_no_overlap(
    monomer_a,
    monomer_b,
    rotation_angle,
    axis_vector,
    initial_separation,
    min_allowed_distance,
    separation_increment,
    max_separation,
):
    """
    Builds a dimer by rotating monomer_b around the given axis by the
    given angle, then stacking it above monomer_a. If the two molecules
    are too close (overlapping), the stacking separation is increased
    step by step until the minimum distance constraint is satisfied.

    Parameters
    ----------
    monomer_a : ASE Atoms
        Bottom molecule, kept fixed.
    monomer_b : ASE Atoms
        Top molecule, rotated and shifted upward.
    rotation_angle : float
        Misorientation angle in degrees (0-90), matching your bicrystal
        convention.
    axis_vector : array-like of length 3
        Rotation axis (a-axis, b-axis, or plane normal).
    initial_separation : float
        Starting vertical (z) separation in Angstrom, before overlap check.
    min_allowed_distance : float
        Minimum allowed distance between any atom pair across the two
        molecules, in Angstrom.
    separation_increment : float
        Step size used to increase separation if overlap is detected.
    max_separation : float
        Safety cap. If exceeded without resolving overlap, raises an error
        so you notice something is wrong rather than silently producing a
        bad structure.

    Returns
    -------
    dimer : ASE Atoms
        Combined, overlap-free dimer structure.
    final_separation : float
        The separation (Angstrom) actually used, after any adjustment.
    """
    # Step 1: rotate the top monomer around the specified axis
    top = rotate_molecule_around_axis(monomer_b, rotation_angle, axis_vector)

    # Step 2: iteratively increase separation until no overlap remains
    separation = initial_separation

    while separation <= max_separation:
        top_shifted = top.copy()
        top_shifted.translate(np.array([0.0, 0.0, separation]))

        min_dist = min_interatomic_distance(monomer_a, top_shifted)

        if min_dist >= min_allowed_distance:
            # Safe: no overlap, we can stop here
            dimer = monomer_a.copy()
            dimer += top_shifted
            return dimer, separation

        separation += separation_increment

    # If we get here, we never resolved the overlap within the safety cap
    raise ValueError(
        f"Could not find overlap-free separation within max_separation="
        f"{max_separation} A for rotation_angle={rotation_angle}, "
        f"axis={axis_vector}. Check your monomer geometry or increase "
        f"max_stack_separation in the config."
    )

def apply_random_slip(molecule, slip_range, rng):
    """
    Applies a small random in-plane (x, y) shift to a molecule, simulating
    local disorder at a grain boundary interface beyond pure rotation.

    Parameters
    ----------
    molecule : ASE Atoms
    slip_range : tuple (low, high)
        Range (Angstrom) to sample the random x and y shift from.
    rng : numpy.random.Generator
        Random number generator (pass one in for reproducibility).
    """
    shifted = molecule.copy()
    dx = rng.uniform(slip_range[0], slip_range[1])
    dy = rng.uniform(slip_range[0], slip_range[1])
    shifted.translate([dx, dy, 0.0])
    return shifted


def apply_thermal_jitter(molecule, std_angstrom, rng, max_displacement=0.15):
    """
    Applies small random displacements to every atom, simulating
    room-temperature vibrational disorder, with a hard cap to prevent
    unphysical bond distortion from rare large random draws.
    """
    jittered = molecule.copy()
    displacements = rng.normal(loc=0.0, scale=std_angstrom, size=jittered.positions.shape)
    displacements = np.clip(displacements, -max_displacement, max_displacement)
    jittered.positions += displacements
    return jittered