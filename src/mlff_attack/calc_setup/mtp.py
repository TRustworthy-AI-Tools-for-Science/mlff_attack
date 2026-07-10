"""MTP calculator setup through MLIP-3."""

import logging
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

import numpy as np
from ase.calculators.calculator import FileIOCalculator
from ase.data import atomic_numbers

from mlff_attack.calc_setup.calculator_class import MLFFCalc

logger = logging.getLogger(__name__)


class MTPCalculator(FileIOCalculator):
    """ASE calculator for MLIP-3 .almtp potentials."""

    implemented_properties = ["energy", "forces"]
    name = "mtp"

    def __init__(
        self,
        model_path,
        elements_path=None,
        mlp_command="mlp",
    ):
        self.model_path = Path(model_path).expanduser().resolve()
        self.check_model_file()

        if elements_path is None:
            elements_path = Path(f"{self.model_path}.elements")

        self.elements_path = Path(elements_path).expanduser().resolve()
        self.check_elements_file()

        self.elements = self.elements_path.read_text(encoding="utf-8").split()
        self.check_elements()

        self.element_types = {}
        for index, element in enumerate(self.elements):
            self.element_types[element] = index

        self.mlp_command = shutil.which(mlp_command)
        if self.mlp_command is None:
            raise FileNotFoundError(
                "Could not find MLIP-3 executable 'mlp'. "
                "Activate env-mtp first."
            )

        self.workdir = tempfile.TemporaryDirectory(prefix="mlff_attack_mtp_")
        self.input_cfg = Path(self.workdir.name) / "input.cfg"
        self.output_cfg = self.input_cfg

        super().__init__(
            label="mtp",
            command=self.mlp_command,
        )

    def check_model_file(self):
        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"MTP model does not exist: {self.model_path}"
            )

        if self.model_path.suffix.lower() != ".almtp":
            raise ValueError("MTP model path must end with '.almtp'")

    def check_elements_file(self):
        if not self.elements_path.is_file():
            raise FileNotFoundError(
                f"MTP element mapping does not exist: {self.elements_path}"
            )

    def check_elements(self):
        if not self.elements:
            raise ValueError("MTP element mapping is empty")

        if len(set(self.elements)) != len(self.elements):
            raise ValueError("MTP element mapping contains duplicate elements")

        for element in self.elements:
            if element not in atomic_numbers:
                raise ValueError(f"Invalid element in MTP mapping: {element}")

        species_count = self.read_species_count()
        if len(self.elements) != species_count:
            raise ValueError(
                f"pot.almtp expects {species_count} species, "
                f"but {self.elements_path.name} contains {len(self.elements)}"
            )

    def read_species_count(self):
        pattern = re.compile(r"^\s*species_count\s*=\s*(\d+)\s*$")

        with self.model_path.open(
            "r",
            encoding="utf-8",
            errors="replace",
        ) as model_file:
            for line in model_file:
                match = pattern.match(line)
                if match:
                    return int(match.group(1))

        raise ValueError("Could not find species_count in MTP model")

    def write_input(
        self,
        atoms,
        properties=None,
        system_changes=None,
    ):
        if not np.all(atoms.get_pbc()):
            raise ValueError("MTP requires periodic boundary conditions")

        cell = np.asarray(atoms.cell.array, dtype=float)
        if cell.shape != (3, 3) or abs(np.linalg.det(cell)) < 1.0e-12:
            raise ValueError("MTP requires a valid three-dimensional cell")

        symbols = atoms.get_chemical_symbols()
        positions = atoms.get_positions()

        for symbol in symbols:
            if symbol not in self.element_types:
                raise ValueError(
                    f"MTP model does not support element '{symbol}'. "
                    f"Supported elements: {', '.join(self.elements)}"
                )

        lines = [
            "BEGIN_CFG",
            " Size",
            f"    {len(atoms)}",
            " Supercell",
        ]

        for vector in cell:
            lines.append(
                "    " + " ".join(f"{value:.16e}" for value in vector)
            )

        lines.append(" AtomData:  id type cartes_x cartes_y cartes_z")

        atom_id = 1
        for symbol, position in zip(symbols, positions):
            atom_type = self.element_types[symbol]
            lines.append(
                f"    {atom_id} {atom_type} "
                f"{position[0]:.16e} "
                f"{position[1]:.16e} "
                f"{position[2]:.16e}"
            )
            atom_id += 1

        lines.append("END_CFG")
        lines.append("")

        self.input_cfg.write_text(
            "\n".join(lines),
            encoding="utf-8",
        )

    def execute(self):
        command = [
            self.mlp_command,
            "calculate_efs",
            str(self.model_path),
            self.input_cfg.name,
        ]

        result = subprocess.run(
            command,
            cwd=self.workdir.name,
            capture_output=True,
            text=True,
            check=False,
        )

        if result.returncode != 0:
            raise RuntimeError(
                "MLIP calculate_efs failed:\n"
                + result.stdout
                + result.stderr
            )

        if not self.input_cfg.is_file():
            raise RuntimeError(
                "MLIP calculate_efs finished but input.cfg is missing.\n"
                "Command:\n"
                + " ".join(command)
                + "\n\nSTDOUT:\n"
                + result.stdout
                + "\nSTDERR:\n"
                + result.stderr
            )

    def get_next_value(self, lines, start):
        for index in range(start, len(lines)):
            value = lines[index].strip()
            if value:
                return value
        raise RuntimeError("Unexpected end of MLIP output")

    def read_results(self):
        if not self.output_cfg.is_file():
            raise RuntimeError("MLIP output file was not created")

        lines = self.output_cfg.read_text(
            encoding="utf-8",
            errors="replace",
        ).splitlines()

        energy = None
        forces = []

        for line_number, line in enumerate(lines):
            stripped = line.strip()

            if stripped == "Energy":
                energy_line = self.get_next_value(lines, line_number + 1)
                energy = float(energy_line.split()[0])

            if stripped.startswith("AtomData:") and "fx" in stripped:
                columns = stripped.split(":", maxsplit=1)[1].split()
                id_column = columns.index("id")
                fx_column = columns.index("fx")
                fy_column = columns.index("fy")
                fz_column = columns.index("fz")

                row_number = line_number + 1
                while len(forces) < len(self.atoms):
                    row = lines[row_number].strip()
                    row_number += 1

                    if not row:
                        continue

                    values = row.split()
                    atom_id = int(values[id_column])
                    force = [
                        float(values[fx_column]),
                        float(values[fy_column]),
                        float(values[fz_column]),
                    ]
                    forces.append((atom_id, force))

        if energy is None:
            raise RuntimeError("Could not read energy from MLIP output")

        if len(forces) != len(self.atoms):
            raise RuntimeError("Could not read forces from MLIP output")

        ordered_forces = np.zeros((len(self.atoms), 3), dtype=np.float64)
        for atom_id, force in forces:
            ordered_forces[atom_id - 1] = force

        self.results = {
            "energy": float(energy),
            "forces": ordered_forces,
        }


class MTPCalcSetup(MLFFCalc):
    """Setup class for MTP."""

    def setup(self, atoms):
        if self.device != "cpu":
            raise ValueError("MTP only supports --device cpu")

        if self.dtype_str != "float64":
            raise ValueError("MTP only supports --dtype float64")

        self.set_seed()

        if self.verbose:
            logger.info("[INFO] Loading MTP model: %s", self.model_path)

        atoms.calc = MTPCalculator(self.model_path)

        self.set_seed()
        return atoms
