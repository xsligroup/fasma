from dataclasses import dataclass
import numpy as np
import h5py


@dataclass(frozen=True)
class BasicData:
    atom_list: list
    scf_type: str
    n_basis: int
    n_primitive_gaussian: int
    n_alpha_electron: int
    n_beta_electron: int
    n_electron: int
    n_mo: int
    n_ao: int
    homo: int
    lumo: int

    def data_to_hdf5(self, file):
        group = file.create_group("basic_data")
        group.create_dataset("atom_list", data=self.atom_list, dtype=h5py.special_dtype(vlen=str))
        group.attrs['scf_type'] = self.scf_type
        group.attrs['n_basis'] = self.n_basis
        group.attrs['n_primitive_gaussian'] = self.n_primitive_gaussian
        group.attrs['n_alpha_electron'] = self.n_alpha_electron
        group.attrs['n_beta_electron'] = self.n_beta_electron
        group.attrs['n_electron'] = self.n_electron
        group.attrs['n_mo'] = self.n_mo
        group.attrs['n_ao'] = self.n_ao
        group.attrs['homo'] = self.homo
        group.attrs['lumo'] = self.lumo


def hdf5_to_data(group):
    return BasicData(atom_list=list(group['atom_list'].asstr()[:]), scf_type=group.attrs['scf_type'], n_basis=group.attrs['n_basis'],
                     n_primitive_gaussian=group.attrs['n_primitive_gaussian'], n_alpha_electron=group.attrs['n_alpha_electron'],
                     n_beta_electron=group.attrs['n_beta_electron'], n_electron=group.attrs['n_electron'],
                     n_mo=group.attrs['n_mo'], n_ao=group.attrs['n_ao'], homo=group.attrs['homo'], lumo=group.attrs['lumo'])
