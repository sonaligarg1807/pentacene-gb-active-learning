import yaml
from pathlib import Path
import numpy as np
from mace.calculators import MACECalculator  # same as you used for eval

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "committee_config.yaml"


class MACECommittee:
    def __init__(self, config_path: Path = CONFIG_PATH):
        # 1. Load YAML
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        self.name = config["name"]
        self.device = config.get("device", "cpu")
        self.output_dir = Path(config["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 2. Build calculators for each model
        self.members = []
        for entry in config["models"]:
            model_type = entry["type"]
            model_path = entry["path"]

            calc = MACECalculator(
                model_paths=[model_path],
                device=self.device,
            )
            self.members.append({"type": model_type, "calculator": calc})

    def predict_single(self, atoms):
        """Return mean and std for energy and forces for one ASE Atoms."""
        # Run all models
        energies = []
        forces = []

        for member in self.members:
            calc = member["calculator"]
            atoms.set_calculator(calc)
            energy = atoms.get_potential_energy()
            force = atoms.get_forces()

            energies.append(energy)
            forces.append(force)

        energies = np.array(energies)              # shape: (n_models,)
        forces = np.stack(forces, axis=0)          # shape: (n_models, n_atoms, 3)

        mean_energy = energies.mean()
        std_energy = energies.std()

        mean_forces = forces.mean(axis=0)          # (n_atoms, 3)
        std_forces = forces.std(axis=0)            # (n_atoms, 3)

        return {
            "mean_energy": mean_energy,
            "std_energy": std_energy,
            "mean_forces": mean_forces,
            "std_forces": std_forces,
        }

    def predict_batch(self, atoms_list):
        """Apply predict_single to a list of ASE Atoms."""
        results = []
        for atoms in atoms_list:
            results.append(self.predict_single(atoms))
        return results