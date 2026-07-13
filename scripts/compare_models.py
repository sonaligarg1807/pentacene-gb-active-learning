import json
from pathlib import Path

import numpy as np
import yaml
from ase.io import read
from mace.calculators import MACECalculator
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "model_comparison_config.yaml"


def load_config():
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Config file not found: {CONFIG_PATH}")

    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)

    required_keys = ["valid_file", "models", "device", "output_dir"]
    missing = [k for k in required_keys if k not in config]
    if missing:
        raise KeyError(f"Missing required config keys: {missing}")

    if not config["models"]:
        raise ValueError("No models specified under 'models' in config.")

    return config


def evaluate_model(model_path, valid_file, device):
    configs = read(valid_file, index=":")
    calc = MACECalculator(model_paths=str(model_path), device=device)

    ref_energies_per_atom, pred_energies_per_atom = [], []
    ref_forces, pred_forces = [], []

    for atoms in configs:
        n_atoms = len(atoms)

        if "REF_energy" not in atoms.info or "REF_forces" not in atoms.arrays:
            print(f"WARNING: skipping a frame missing REF_energy/REF_forces")
            continue

        ref_e = atoms.info["REF_energy"] / n_atoms
        ref_f = atoms.arrays["REF_forces"]

        atoms.calc = calc
        pred_e = atoms.get_potential_energy() / n_atoms
        pred_f = atoms.get_forces()

        ref_energies_per_atom.append(ref_e)
        pred_energies_per_atom.append(pred_e)
        ref_forces.append(ref_f)
        pred_forces.append(pred_f)

    ref_e_arr = np.array(ref_energies_per_atom)
    pred_e_arr = np.array(pred_energies_per_atom)
    ref_f_arr = np.concatenate([f.flatten() for f in ref_forces])
    pred_f_arr = np.concatenate([f.flatten() for f in pred_forces])

    rmse_e = np.sqrt(np.mean((ref_e_arr - pred_e_arr) ** 2)) * 1000
    rmse_f = np.sqrt(np.mean((ref_f_arr - pred_f_arr) ** 2)) * 1000
    rel_f_rmse = (
        np.sqrt(np.mean((ref_f_arr - pred_f_arr) ** 2))
        / np.sqrt(np.mean(ref_f_arr ** 2))
        * 100
    )

    return {
        "rmse_e_mev_per_atom": float(rmse_e),
        "rmse_f_mev_per_A": float(rmse_f),
        "relative_f_rmse_pct": float(rel_f_rmse),
        "ref_e": ref_e_arr,
        "pred_e": pred_e_arr,
        "ref_f": ref_f_arr,
        "pred_f": pred_f_arr,
    }


def make_parity_plots(results, output_dir, plot_colors):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for name, r in results.items():
        color = plot_colors.get(name, None)
        axes[0].scatter(r["ref_e"], r["pred_e"], s=10, alpha=0.6, label=name, color=color)
    all_e = np.concatenate([r["ref_e"] for r in results.values()])
    lims_e = [all_e.min(), all_e.max()]
    axes[0].plot(lims_e, lims_e, "k--", linewidth=1)
    axes[0].set_xlabel("Reference Energy per atom [eV]")
    axes[0].set_ylabel("Predicted Energy per atom [eV]")
    axes[0].set_title("Energy Parity: Scratch vs. Fine-Tuned")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    for name, r in results.items():
        color = plot_colors.get(name, None)
        axes[1].scatter(r["ref_f"], r["pred_f"], s=5, alpha=0.4, label=name, color=color)
    all_f = np.concatenate([r["ref_f"] for r in results.values()])
    lims_f = [all_f.min(), all_f.max()]
    axes[1].plot(lims_f, lims_f, "k--", linewidth=1)
    axes[1].set_xlabel("Reference Force [eV/A]")
    axes[1].set_ylabel("Predicted Force [eV/A]")
    axes[1].set_title("Force Parity: Scratch vs. Fine-Tuned")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plot_path = Path(output_dir) / "parity_comparison.png"
    plt.savefig(plot_path, dpi=150)
    print(f"Saved parity plot to {plot_path}")


def main():
    config = load_config()

    valid_file = config["valid_file"]
    if not Path(valid_file).exists():
        raise FileNotFoundError(f"Validation file not found: {valid_file}")

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    device = config["device"]
    plot_colors = config.get("plot_colors", {})

    results = {}
    for name, model_path in config["models"].items():
        if not Path(model_path).exists():
            print(f"WARNING: model file not found for '{name}': {model_path}")
            continue
        print(f"\nEvaluating '{name}' model: {model_path}")
        results[name] = evaluate_model(model_path, valid_file, device)
        r = results[name]
        print(f"  RMSE_E: {r['rmse_e_mev_per_atom']:.2f} meV/atom")
        print(f"  RMSE_F: {r['rmse_f_mev_per_A']:.2f} meV/A")
        print(f"  Relative F RMSE: {r['relative_f_rmse_pct']:.2f} %")

    if not results:
        raise RuntimeError("No models were successfully evaluated. Check config paths.")

    summary = {
        name: {
            "rmse_e_mev_per_atom": r["rmse_e_mev_per_atom"],
            "rmse_f_mev_per_A": r["rmse_f_mev_per_A"],
            "relative_f_rmse_pct": r["relative_f_rmse_pct"],
        }
        for name, r in results.items()
    }
    summary_path = output_dir / "comparison_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary to {summary_path}")

    print("\n=== COMPARISON TABLE ===")
    print(f"{'Model':<12} {'RMSE_E (meV/atom)':<20} {'RMSE_F (meV/A)':<18} {'Rel. F RMSE (%)':<16}")
    for name, r in results.items():
        print(
            f"{name:<12} {r['rmse_e_mev_per_atom']:<20.2f} "
            f"{r['rmse_f_mev_per_A']:<18.2f} {r['relative_f_rmse_pct']:<16.2f}"
        )

    make_parity_plots(results, output_dir, plot_colors)
    print(f"\nAll outputs saved to {output_dir}")


if __name__ == "__main__":
    main()