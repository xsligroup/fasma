from fasma.core.dataclasses.data import electron
from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class PopData:
    ao_matrix: np.ndarray
    overlap_matrix: np.ndarray
    electron_data: electron.ElectronData
    beta_electron_data: Optional[electron.BetaData] = None

    def add_beta_electron_data(self, data: electron.BetaData):
        if self.beta_electron_data is None:
            self.beta_electron_data = data

    def data_to_hdf5(self, file):
        group = file.create_group("pop_data")
        group.create_dataset("ao_matrix", data=self.ao_matrix.astype('O'))
        group.create_dataset("overlap_matrix", data=self.overlap_matrix)
        self.electron_data.data_to_hdf5(group)
        if self.beta_electron_data is not None:
            self.beta_electron_data.data_to_hdf5(group)


def hdf5_to_data(group):
    electron_data = electron.ElectronData(density_matrix=group["electron_data/density_matrix"][:, :],
                                          mo_coefficient_matrix=group["electron_data/mo_coefficient_matrix"][:, :],
                                          eigenvalues=group["electron_data/eigenvalues"][:],
                                          ao_projection_matrix=group["electron_data/ao_projection_matrix"][:, :])
    pop_data = PopData(ao_matrix=group["ao_matrix"][:, :].astype("<U12"),
                       overlap_matrix=group["overlap_matrix"][:, :], electron_data=electron_data)
    if "beta_electron_data" in group.keys():
        beta_keys = group["beta_electron_data"].keys()
        if "mo_coefficient_matrix" not in beta_keys:
            mo_coefficient_matrix = None
        else:
            mo_coefficient_matrix = group["beta_electron_data/mo_coefficient_matrix"][:, :]
        if "eigenvalues" not in beta_keys:
            eigenvalues = None
        else:
            eigenvalues = group["beta_electron_data/eigenvalues"][:]
        if "ao_projection_matrix" not in beta_keys:
            ao_projection_matrix = None
        else:
            ao_projection_matrix = group["beta_electron_data/ao_projection_matrix"][:, :]
        beta_electron_data = electron.ElectronData(density_matrix=group["beta_electron_data/density_matrix"][:, :],
                                                   mo_coefficient_matrix=mo_coefficient_matrix,
                                                   eigenvalues=eigenvalues,
                                                   ao_projection_matrix=ao_projection_matrix)
        pop_data.add_beta_electron_data(beta_electron_data)
    return pop_data
