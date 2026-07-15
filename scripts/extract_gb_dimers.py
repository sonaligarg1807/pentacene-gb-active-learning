"""
extract_gb_dimers.py

Extracts GB-region dimer structures from equilibrated GROMACS
bicrystal trajectories, using resid pairs listed in each system's
dimer.ndx file. Writes one multi-frame .xyz per system, where each
frame corresponds to one (trajectory_frame, dimer) combination.

INPUT FILES NEEDED (per system, listed in config):
- .tpr topology file
- .xtc trajectory file
- dimer.ndx (plain text, two resids per line, no header)

OUTPUT FILES PRODUCED:
- One .xyz file per system (as specified in output_xyz)

HOW TO RUN:
    python scripts/extract_gb_dimers.py
"""

import yaml
from pathlib import Path
import MDAnalysis as mda
from MDAnalysis.topology.guessers import guess_types

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "gb_dimer_extraction_config.yaml"


def read_dimer_pairs(ndx_path):
    pairs = []
    with open(ndx_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            resid1, resid2 = map(int, line.split())
            pairs.append((resid1, resid2))
    return pairs


def extract_system(system_cfg, stride):
    name = system_cfg["name"]
    tpr = system_cfg["tpr_file"]
    xtc = system_cfg["xtc_file"]
    ndx = system_cfg["ndx_file"]
    out_path = Path(system_cfg["output_xyz"])
    out_path.parent.mkdir(parents=True, exist_ok=True)

    dimer_pairs = read_dimer_pairs(ndx)
    u = mda.Universe(tpr, xtc)
    guessed_elements = guess_types(u.atoms.names)
    u.add_TopologyAttr("elements", guessed_elements)

    n_written = 0
    with open(out_path, "w") as fout:
        for frame_idx, ts in enumerate(u.trajectory[::stride]):
            for resid1, resid2 in dimer_pairs:
                sel = u.select_atoms(f"resid {resid1} {resid2}")
                if sel.n_atoms == 0:
                    continue

                fout.write(f"{sel.n_atoms}\n")
                fout.write(
                    f"system={name} frame={frame_idx} "
                    f"dimer_resids={resid1},{resid2}\n"
                )
                for atom in sel:
                    fout.write(
                        f"{atom.element if atom.element else atom.name[0]} "
                        f"{atom.position[0]:.6f} "
                        f"{atom.position[1]:.6f} "
                        f"{atom.position[2]:.6f}\n"
                    )
                n_written += 1

    print(f"[{name}] Wrote {n_written} dimer frames to {out_path}")


def main():
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)

    stride = config.get("stride", 1)

    for system_cfg in config["systems"]:
        extract_system(system_cfg, stride)


if __name__ == "__main__":
    main()