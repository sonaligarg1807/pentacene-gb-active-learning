"""
select_high_uncertainty_dimers.py

Selects the top-percentile most force-uncertain GB dimer structures
per system (based on committee mean_std_force), and writes them out
as .xyz files ready for xTB labeling. Also writes one combined .xyz
across all systems.

INPUT FILES NEEDED (per system, listed in config):
- committee stats JSON (from run_committee_on_xyz.py)
- corresponding subset .xyz file (same frame order as JSON index)

OUTPUT FILES PRODUCED:
- Per-system top-percentile .xyz files
- One combined .xyz across all systems

HOW TO RUN:
    python scripts/select_high_uncertainty_dimers.py
"""

import json
import yaml
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "xTB_selection_config.yaml"


def load_xyz_frames(xyz_path):
    """Return a list of frame strings (n_atoms line + comment + coords), indexed like ASE would."""
    frames = []
    with open(xyz_path, "r") as f:
        while True:
            line = f.readline()
            if not line:
                break
            n_atoms = int(line.strip())
            comment = f.readline()
            coords = [f.readline() for _ in range(n_atoms)]
            frames.append(line + comment + "".join(coords))
    return frames


def select_top_percentile(stats_json_path, percentile):
    with open(stats_json_path, "r") as f:
        stats = json.load(f)

    stats_sorted = sorted(stats, key=lambda d: d["mean_std_force"], reverse=True)
    n_select = max(1, int(len(stats_sorted) * percentile))
    selected = stats_sorted[:n_select]
    selected_indices = sorted(d["index"] for d in selected)

    return selected_indices, selected


def process_system(system_cfg, top_percentile, combined_frames):
    name = system_cfg["name"]
    stats_json_path = system_cfg["stats_json"]
    xyz_path = system_cfg["xyz_file"]
    output_xyz = Path(system_cfg["output_xyz"])
    output_xyz.parent.mkdir(parents=True, exist_ok=True)

    selected_indices, selected_stats = select_top_percentile(stats_json_path, top_percentile)
    all_frames = load_xyz_frames(xyz_path)

    with open(output_xyz, "w") as fout:
        for idx in selected_indices:
            frame_str = all_frames[idx]
            # Tag the comment line with the uncertainty value for traceability
            lines = frame_str.split("\n", 2)
            n_atoms_line = lines[0]
            comment_line = lines[1]
            rest = lines[2] if len(lines) > 2 else ""

            std_force_val = next(
                d["mean_std_force"] for d in selected_stats if d["index"] == idx
            )
            new_comment = f"{comment_line} mean_std_force={std_force_val:.6f}"

            fout.write(f"{n_atoms_line}\n{new_comment}\n{rest}")
            combined_frames.append(f"{n_atoms_line}\n{new_comment}\n{rest}")

    print(f"[{name}] Selected {len(selected_indices)} / {len(all_frames)} "
          f"structures (top {int(top_percentile * 100)}%) -> {output_xyz}")


def main():
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)

    top_percentile = config.get("top_percentile", 0.20)
    combined_frames = []

    for system_cfg in config["systems"]:
        process_system(system_cfg, top_percentile, combined_frames)

    combined_path = Path(config["combined_output_xyz"])
    combined_path.parent.mkdir(parents=True, exist_ok=True)
    with open(combined_path, "w") as fout:
        fout.writelines(combined_frames)

    print(f"\nWrote combined selection ({len(combined_frames)} structures) to {combined_path}")


if __name__ == "__main__":
    main()