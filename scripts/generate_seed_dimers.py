"""
generate_seed_dimers.py (expanded)

Generates an expanded seed dataset of pentacene dimers by combining:
- rotation axis (a_axis, b_axis, normal_ac, normal_bc)
- rotation angle (0-90 deg, 5 deg steps)
- multiple pi-stack separations per combination
- small random slip and thermal jitter for realistic disorder

INPUT FILES NEEDED:
- data/seed_structures/pentacene_monomer.xyz
- configs/seed_generation.yaml

OUTPUT FILES PRODUCED:
- data/seed_structures/generated_dimers/dimer_<axis>_<angle>deg_sep<N>_<variant>.xyz
- data/seed_structures/seed_dimers_manifest.csv

HOW TO RUN:
    python scripts/generate_seed_dimers.py
"""

import os
import sys
import csv
import yaml
import numpy as np
from ase.io import write

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.dimer_utils import (
    load_monomer,
    center_molecule,
    build_dimer_no_overlap,
    apply_random_slip,
    apply_thermal_jitter,
)


def load_config(config_path="configs/seed_generation.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    config = load_config()
    os.makedirs(config["output_dir"], exist_ok=True)
    rng = np.random.default_rng(config["random_seed"])

    print(f"Loading monomer from {config['input_monomer']} ...")
    monomer_template = load_monomer(config["input_monomer"])
    monomer_template = center_molecule(monomer_template)
    print(f"Loaded monomer with {len(monomer_template)} atoms.")

    angle_cfg = config["angle_range"]
    angles = list(range(angle_cfg["start"], angle_cfg["stop"] + 1, angle_cfg["step"]))
    axes = config["rotation_axes"]
    offsets = config["extra_separation_offsets"]

    print(f"Angles: {len(angles)} | Axes: {len(axes)} | Separation offsets: {len(offsets)}")
    print(f"Expected total (before jitter variants): "
          f"{len(angles) * len(axes) * len(offsets)}")

    manifest_rows = []
    count = 0
    skipped = 0

    for axis_name, axis_vector in axes.items():
        for angle in angles:
            monomer_a = monomer_template.copy()
            monomer_b = monomer_template.copy()

            try:
                base_dimer, base_separation = build_dimer_no_overlap(
                    monomer_a, monomer_b,
                    rotation_angle=angle,
                    axis_vector=np.array(axis_vector, dtype=float),
                    initial_separation=config["initial_stack_separation"],
                    min_allowed_distance=config["min_allowed_distance"],
                    separation_increment=config["separation_increment"],
                    max_separation=config["max_stack_separation"],
                )
            except ValueError as e:
                print(f"SKIPPED axis={axis_name}, angle={angle}: {e}")
                skipped += 1
                continue

            # For each extra separation offset, build a variant
            for offset in offsets:
                monomer_a2 = monomer_template.copy()
                monomer_b2 = monomer_template.copy()

                try:
                    dimer, sep_used = build_dimer_no_overlap(
                        monomer_a2, monomer_b2,
                        rotation_angle=angle,
                        axis_vector=np.array(axis_vector, dtype=float),
                        initial_separation=base_separation + offset,
                        min_allowed_distance=config["min_allowed_distance"],
                        separation_increment=config["separation_increment"],
                        max_separation=config["max_stack_separation"],
                    )
                except ValueError:
                    continue  # skip this offset if it still can't resolve

                # Optionally apply random slip
                dimer = apply_random_slip(dimer, config["slip_range"], rng)

                # Optionally apply thermal jitter to a fraction of structures
                variant_tag = "base"
                if rng.uniform() < config["jitter_fraction"]:
                    dimer = apply_thermal_jitter(dimer, config["jitter_std_angstrom"], rng)
                    variant_tag = "jitter"

                filename = (f"dimer_{axis_name}_{angle:03d}deg_"
                            f"sep{sep_used:.2f}_{variant_tag}_{count:04d}.xyz")
                filepath = os.path.join(config["output_dir"], filename)
                write(filepath, dimer)

                manifest_rows.append({
                    "filename": filename,
                    "rotation_axis": axis_name,
                    "rotation_angle_deg": angle,
                    "stack_separation_A": round(sep_used, 3),
                    "variant": variant_tag,
                })

                count += 1

        print(f"Progress: axis {axis_name} done. Total structures so far: {count}")

    manifest_path = config["manifest_file"]
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=manifest_rows[0].keys())
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"\nDone. {count} dimer structures saved to {config['output_dir']}")
    print(f"Skipped combinations (overlap unresolved): {skipped}")
    print(f"Manifest saved to {manifest_path}")


if __name__ == "__main__":
    main()