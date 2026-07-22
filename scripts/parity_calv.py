# ==========================================================
# Parity data generation: MACE (round0 vs round1) vs xTB ground truth
# (Calculation + saving only -- plotting done separately, e.g. in Jupyter)
# ==========================================================
import numpy as np
import pandas as pd
from ase.io import read
from mace.calculators import MACECalculator
import os

# ============ PLACEHOLDERS: update these paths ============
HELD_OUT_XYZ = "/home/sgarg/git-repo/pentacene-gb-active-learning/data/labeled_data/gb_al_round1_valid.xyz"

ROUND0_MODEL_PATHS = [
    "/home/sgarg/git-repo/pentacene-gb-active-learning/output/models/finetune_v1/pentacene_finetune_v1.model",
    "/home/sgarg/git-repo/pentacene-gb-active-learning/output/models/finetune_v1_seed99/pentacene_finetune_v1.model",
    "/home/sgarg/git-repo/pentacene-gb-active-learning/output/models/finetune_v1_seed123/pentacene_finetune_v1.model",
]
ROUND1_MODEL_PATHS = [
    "/home/sgarg/git-repo/pentacene-gb-active-learning/output/gb_al_round1/retrain_v1_seed142/gb_al_round1.model",
    "/home/sgarg/git-repo/pentacene-gb-active-learning/output/gb_al_round1/retrain_v1_seed299/gb_al_round1.model",
    "/home/sgarg/git-repo/pentacene-gb-active-learning/output/gb_al_round1/retrain_v1_seed523/gb_al_round1.model",
]

DEVICE = "cpu"  # or "cuda"
OUTPUT_DIR = "/home/sgarg/git-repo/pentacene-gb-active-learning/output/parity_plots"

# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---- Load held-out structures ----
atoms_list = read(HELD_OUT_XYZ, index=":")
atoms_list = [a for a in atoms_list if "REF_energy" in a.info and "REF_forces" in a.arrays]
print(f"Using {len(atoms_list)} labeled structures")

# xTB ground truth: assumed stored in atoms.info["REF_energy"] and atoms.arrays["REF_forces"]
xtb_energies = np.array([a.info["REF_energy"] for a in atoms_list])
xtb_forces = np.concatenate([a.arrays["REF_forces"].flatten() for a in atoms_list])
n_atoms_per_struct = np.array([len(a) for a in atoms_list])


def get_committee_predictions(atoms_list, model_paths):
    calcs = [MACECalculator(model_paths=[p], device=DEVICE) for p in model_paths]
    energies_per_model = []
    forces_per_model = []
    for calc in calcs:
        e_list, f_list = [], []
        for atoms in atoms_list:
            atoms_copy = atoms.copy()
            atoms_copy.calc = calc
            e_list.append(atoms_copy.get_potential_energy())
            f_list.append(atoms_copy.get_forces().flatten())
        energies_per_model.append(np.array(e_list))
        forces_per_model.append(np.concatenate(f_list))
    energies_per_model = np.array(energies_per_model)
    forces_per_model = np.array(forces_per_model)
    return energies_per_model.mean(axis=0), forces_per_model.mean(axis=0)


print("Running round0 committee predictions...")
mace_e_r0, mace_f_r0 = get_committee_predictions(atoms_list, ROUND0_MODEL_PATHS)

print("Running round1 committee predictions...")
mace_e_r1, mace_f_r1 = get_committee_predictions(atoms_list, ROUND1_MODEL_PATHS)

xtb_e_per_atom = xtb_energies / n_atoms_per_struct
mace_e_r0_per_atom = mace_e_r0 / n_atoms_per_struct
mace_e_r1_per_atom = mace_e_r1 / n_atoms_per_struct

df_energy = pd.DataFrame({
    "xtb_energy_per_atom": xtb_e_per_atom,
    "mace_r0_energy_per_atom": mace_e_r0_per_atom,
    "mace_r1_energy_per_atom": mace_e_r1_per_atom,
})
df_energy.to_csv(f"{OUTPUT_DIR}/energy_parity_data.csv", index=False)

df_forces = pd.DataFrame({
    "xtb_force": xtb_forces,
    "mace_r0_force": mace_f_r0,
    "mace_r1_force": mace_f_r1,
})
df_forces.to_csv(f"{OUTPUT_DIR}/force_parity_data.csv", index=False)


def compute_metrics(true, pred):
    rmse = np.sqrt(np.mean((true - pred) ** 2))
    mae = np.mean(np.abs(true - pred))
    return rmse, mae


e_rmse_r0, e_mae_r0 = compute_metrics(xtb_e_per_atom, mace_e_r0_per_atom)
e_rmse_r1, e_mae_r1 = compute_metrics(xtb_e_per_atom, mace_e_r1_per_atom)
f_rmse_r0, f_mae_r0 = compute_metrics(xtb_forces, mace_f_r0)
f_rmse_r1, f_mae_r1 = compute_metrics(xtb_forces, mace_f_r1)

metrics_df = pd.DataFrame([
    {"round": "round0", "metric": "energy_per_atom_rmse", "value": e_rmse_r0},
    {"round": "round0", "metric": "energy_per_atom_mae", "value": e_mae_r0},
    {"round": "round0", "metric": "force_rmse", "value": f_rmse_r0},
    {"round": "round0", "metric": "force_mae", "value": f_mae_r0},
    {"round": "round1", "metric": "energy_per_atom_rmse", "value": e_rmse_r1},
    {"round": "round1", "metric": "energy_per_atom_mae", "value": e_mae_r1},
    {"round": "round1", "metric": "force_rmse", "value": f_rmse_r1},
    {"round": "round1", "metric": "force_mae", "value": f_mae_r1},
])
metrics_df.to_csv(f"{OUTPUT_DIR}/parity_metrics.csv", index=False)
print(metrics_df.to_string(index=False))

print("\nAll CSVs and metrics written to", OUTPUT_DIR)