from fasma.core.dataclasses.data import spectra, methodology
from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class ExcitationData(spectra.SpectraData):
    n_ground_state: int
    n_excited_state: int
    final_state: int
    n_excitation: int
    n_active_space_mo: int
    n_active_space_electron: int
    active_space_start: int
    active_space_end: int
    delta_diagonal_matrix: np.ndarray = None
    beta_delta_diagonal_matrix: Optional[np.ndarray] = None
    excitation_matrix: Optional[np.ndarray] = None
    methodology_data: Optional[methodology.MethodologyData] = None
    method: Optional[str] = None

    def initialize_active_space(self):
        self.active_space_end = self.active_space_start + self.n_active_space_mo

    def add_excitation_matrix(self, data: np.ndarray):
        if self.excitation_matrix is None:
            self.excitation_matrix = data

    def add_delta_diagonal_matrix(self, data: np.ndarray):
        if self.delta_diagonal_matrix is None:
            self.delta_diagonal_matrix = data

    def add_beta_delta_diagonal_matrix(self, data: np.ndarray):
        if self.beta_delta_diagonal_matrix is None:
            self.beta_delta_diagonal_matrix = data

    def add_methodology_data(self, data: methodology.MethodologyData):
        if self.methodology_data is None:
            self.methodology_data = data

    def data_to_hdf5(self, file):
        group = file.create_group("spectra_data")
        group.attrs['n_ground_state'] = self.n_ground_state
        group.attrs['n_excited_state'] = self.n_excited_state
        group.attrs['final_state'] = self.final_state
        group.attrs['n_excitation'] = self.n_excitation
        group.attrs['n_active_space_mo'] = self.n_active_space_mo
        group.attrs['n_active_space_electron'] = self.n_active_space_electron
        group.attrs['active_space_start'] = self.active_space_start
        group.attrs['active_space_end'] = self.active_space_end
        group.attrs['method'] = self.method
        if self.delta_diagonal_matrix is not None:
            group.create_dataset("delta_diagonal_matrix", data=self.delta_diagonal_matrix)
        if self.beta_delta_diagonal_matrix is not None:
            group.create_dataset("beta_delta_diagonal_matrix", data=self.beta_delta_diagonal_matrix)
        if self.excitation_matrix is not None:
            group.create_dataset("excitation_matrix", data=self.excitation_matrix)
        if self.methodology_data is not None:
            self.methodology_data.data_to_hdf5(group)


@dataclass
class TDData(ExcitationData):
    n_ground_state: int = field(init=False)
    final_state: int = field(init=False)
    n_excitation: int = field(init=False)
    active_space_start: int = field(init=False)
    active_space_end: int = field(init=False)

    def __post_init__(self):
        self.n_ground_state = 1
        self.final_state = self.n_ground_state + self.n_excited_state
        self.n_excitation = self.n_excited_state
        self.active_space_start = 0
        self.initialize_active_space()
        self.method = "TD"


@dataclass
class CASData(ExcitationData):
    n_excited_state: int = field(init=False)
    n_excitation: int = field(init=False)
    active_space_end: int = field(init=False)

    def __post_init__(self):
        self.n_excited_state = self.methodology_data.n_root - self.n_ground_state
        self.n_excitation = int((self.final_state * self.n_ground_state) - (self.n_ground_state * (self.n_ground_state + 1) / 2))
        self.initialize_active_space()
        self.method = "CAS"


def hdf5_to_data(group):
    key_list = list(group.keys())
    group_method = group.attrs["method"]
    if group_method == "CAS":
        methodology_data = methodology.hdf5_to_data(group["methodology_data"])
        excitation_data = CASData(n_ground_state=group.attrs['n_ground_state'], final_state=group.attrs['final_state'],
                                  n_active_space_mo=group.attrs['n_active_space_mo'],
                                  n_active_space_electron=group.attrs['n_active_space_electron'],
                                  active_space_start=group.attrs['active_space_start'],
                                  methodology_data=methodology_data)
    elif group_method == "TD":
        excitation_data = TDData(n_excited_state=group.attrs['n_excited_state'],
                                 n_active_space_mo=group.attrs['n_active_space_mo'],
                                 n_active_space_electron=group.attrs['n_active_space_electron'])
    else:
        excitation_data = ExcitationData(n_ground_state=group.attrs['n_ground_state'],
                                         n_excited_state=group.attrs['n_excited_state'],
                                         final_state=group.attrs['final_state'],
                                         n_excitation=group.attrs['n_excitation'],
                                         n_active_space_mo=group.attrs['n_active_space_mo'],
                                         n_active_space_electron=group.attrs['n_active_space_electron'],
                                         active_space_start=group.attrs['active_space_start'],
                                         active_space_end=group.attrs['active_space_end'])
    if "excitation_matrix" in key_list:
        excitation_data.add_excitation_matrix(group["excitation_matrix"][:, :])
    if "delta_diagonal_matrix" in key_list:
        excitation_data.add_delta_diagonal_matrix(group["delta_diagonal_matrix"][:, :])
    if "beta_delta_diagonal_matrix" in key_list:
        excitation_data.add_beta_delta_diagonal_matrix(group["beta_delta_diagonal_matrix"][:, :])
    return excitation_data

