from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class ElectronData:
    mo_coefficient_matrix: np.ndarray
    eigenvalues: np.ndarray
    density_matrix: Optional[np.ndarray] = None
    ao_projection_matrix: Optional[np.ndarray] = None

    def add_ao_projection_matrix(self, data: np.ndarray):
        if self.ao_projection_matrix is None:
            self.ao_projection_matrix = data

    def data_to_hdf5(self, group):
        electron_group = group.create_group("electron_data")
        electron_group.create_dataset("mo_coefficient_matrix", data=self.mo_coefficient_matrix)
        electron_group.create_dataset("eigenvalues", data=self.eigenvalues)
        if self.density_matrix is not None:
            electron_group.create_dataset("density_matrix", data=self.density_matrix)
        if self.ao_projection_matrix is not None:
            electron_group.create_dataset("ao_projection_matrix", data=self.ao_projection_matrix)


@dataclass
class BetaData(ElectronData):
    mo_coefficient_matrix: Optional[np.ndarray] = None
    eigenvalues: Optional[np.ndarray] = None

    def add_beta_mo_coefficient_matrix(self, data: np.ndarray):
        if self.mo_coefficient_matrix is None:
            self.mo_coefficient_matrix = data

    def add_beta_eigenvalues(self, data: np.ndarray):
        if self.eigenvalues is None:
            self.eigenvalues = data

    def data_to_hdf5(self, group):
        electron_group = group.create_group("beta_electron_data")
        if self.density_matrix is not None:
            electron_group.create_dataset("density_matrix", data=self.density_matrix)
        if self.mo_coefficient_matrix is not None:
            electron_group.create_dataset("mo_coefficient_matrix", data=self.mo_coefficient_matrix)
        if self.eigenvalues is not None:
            electron_group.create_dataset("eigenvalues", data=self.eigenvalues)
        if self.ao_projection_matrix is not None:
            electron_group.create_dataset("ao_projection_matrix", data=self.ao_projection_matrix)
