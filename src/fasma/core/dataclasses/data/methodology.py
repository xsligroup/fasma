from dataclasses import dataclass
from typing import Optional
from abc import ABC
import numpy as np


class MethodologyData(ABC):
    "Methodology Data"

@dataclass
class TDSOCentricData(MethodologyData):
    n_so_excitation: int
    so_excitation_matrix: Optional[np.ndarray] = None


@dataclass
class CASCentricData(MethodologyData):
    n_root: int
    n_excitation_full: int
    switched_orbitals: Optional[np.ndarray] = None

    def add_switched_orbitals(self, data: np.ndarray):
        if self.switched_orbitals is None:
            self.switched_orbitals = data

    def data_to_hdf5(self, group):
        methodology_group = group.create_group("methodology_data")
        methodology_group.attrs['n_root'] = self.n_root
        methodology_group.attrs['n_excitation_full'] = self.n_excitation_full
        if self.switched_orbitals is not None:
            methodology_group.create_dataset("switched_orbitals", data=self.switched_orbitals)


def hdf5_to_data(group):
    if "switched_orbitals" not in group.keys():
        switched_orbitals = None
    else:
        switched_orbitals = group["switched_orbitals"][:, :]
    return CASCentricData(n_root=group.attrs['n_root'], n_excitation_full=group.attrs['n_excitation_full'],
                          switched_orbitals=switched_orbitals)