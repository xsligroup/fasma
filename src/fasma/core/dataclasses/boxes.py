from fasma.core.dataclasses.data import basic, spectra, pop
from fasma.core import df_generators as dfg
from fasma.gaussian import parse_matrices
from dataclasses import dataclass
from typing import Optional
import pandas as pd
import numpy as np


@dataclass
class Box:
    basic_data: basic.BasicData
    spectra_data: Optional[spectra.SpectraData] = None
    pop_data: Optional[pop.PopData] = None

    def add_spectra_data(self, data: spectra.SpectraData):
        if self.spectra_data is None:
            self.spectra_data = data

    def add_pop_data(self, data: pop.PopData):
        if self.pop_data is None:
            self.pop_data = data

    def generate_mo_analysis(self, electron: str = "alpha", full=False):
        if self.pop_data is None:
            raise ValueError(
                "Cannot perform an MO Analysis. Please check that this object has population calculation.")
        if electron == "beta" and self.basic_data.scf_type == "UHF":
            ao_projection_matrix = self.pop_data.beta_electron_data.ao_projection_matrix
            eigenvalues = self.pop_data.beta_electron_data.eigenvalues
        else:
            ao_projection_matrix = self.pop_data.electron_data.ao_projection_matrix
            eigenvalues = self.pop_data.electron_data.eigenvalues
        ao_matrix = self.pop_data.ao_matrix
        ao_df = dfg.get_ao_dataframe(ao_matrix)
        df = dfg.get_mo_dataframe(ao_projection_matrix, 'MO ')
        df = pd.concat([ao_df, df], axis=1)
        df.set_index(
            ['Atom Number', 'Atom Type', 'Principal Quantum Number', 'Subshell', 'Atomic Orbital'], inplace=True)
        if full:
            df_1 = pd.concat([df], keys=[''], names=['Info'])
            info_df = pd.DataFrame(columns=list(df_1.index.names) + list(df_1.columns))
            info_df.set_index(list(df_1.index.names), inplace=True)
            info_df.loc['Energy', :] = eigenvalues
            info_df.loc['Total Sum', :] = df_1.sum().values
            df = pd.concat([info_df, df_1])
        return df

    def generate_mo_transition_analysis(self, electron: str = "alpha"):
        if self.spectra_data is None:
            raise ValueError(
                "Cannot perform an MO Transition Analysis. Please check that this object has an excited state calculation.")
        if electron == "beta":
            delta_diagonal_matrix = self.spectra_data.beta_delta_diagonal_matrix
        else:
            delta_diagonal_matrix = self.spectra_data.delta_diagonal_matrix
        excitation_matrix = self.spectra_data.excitation_matrix
        excitation_df = dfg.get_excitations_dataframe(self.spectra_data.methodology, excitation_matrix)
        summary_df = dfg.get_summary_dataframe(parse_matrices.summarize_matrix(delta_diagonal_matrix))
        df = dfg.get_mo_dataframe(delta_diagonal_matrix, 'AS MO ')
        df = pd.concat([excitation_df, summary_df, df], axis=1)
        df.set_index(['Starting State', 'Ending State'], inplace=True)
        return df

    def generate_merged_mo_transition_analysis(self):
        if self.spectra_data is None:
            raise ValueError(
                "Cannot perform an MO Transition Analysis. Please check that this object has an excited state calculation.")
        alpha_delta_diagonal_matrix = self.spectra_data.delta_diagonal_matrix
        alpha_delta_mo_transition_df = dfg.get_mo_dataframe(alpha_delta_diagonal_matrix, 'AS MO ')
        #alpha_summary_df = dfg.get_summary_dataframe(parse_matrices.summarize_matrix(alpha_delta_diagonal_matrix))

        beta_delta_diagonal_matrix = self.spectra_data.beta_delta_diagonal_matrix
        beta_mo_transition_df = dfg.get_mo_dataframe(beta_delta_diagonal_matrix, 'Beta AS MO ')
        #beta_summary_df = dfg.get_summary_dataframe(parse_matrices.summarize_matrix(beta_delta_diagonal_matrix))

        excitation_matrix = self.spectra_data.excitation_matrix
        excitation_df = dfg.get_excitations_dataframe(self.spectra_data.methodology, excitation_matrix)

        df = pd.concat([excitation_df, alpha_delta_mo_transition_df, beta_mo_transition_df], axis=1)
        df.set_index(['Starting State', 'Ending State'], inplace=True)
        return df

    def generate_ao_transition_analysis(self, electron: str = "alpha"):
        if self.pop_data is None and self.spectra_data is None:
            raise ValueError(
                "Cannot perform an AO Projection Transition Analysis. Please check this object has both a population calculation and an excited state calculation.")
        ao_matrix = self.pop_data.ao_matrix
        ao_matrix = parse_matrices.convert_ao_projection_to_mo_transition(self.spectra_data.n_excitation, ao_matrix)
        ao_df = dfg.get_ao_dataframe(ao_matrix)

        excitation_matrix = self.spectra_data.excitation_matrix
        excitation_matrix = parse_matrices.convert_mo_transition_to_ao_projection(self.basic_data.n_mo, self.spectra_data.n_excitation,  excitation_matrix)
        excitation_df = dfg.get_excitations_dataframe(self.spectra_data.methodology, excitation_matrix)
        ao_transition_matrix = self.generate_ao_transition_matrix(electron)
        summary_df = dfg.get_summary_dataframe(parse_matrices.summarize_matrix(ao_transition_matrix))
        df = dfg.get_mo_dataframe(ao_transition_matrix, 'AS MO ')
        df = pd.concat([excitation_df, ao_df, summary_df, df], axis=1)
        index_list = ['Starting State', 'Ending State', 'Atom Number', 'Atom Type', 'Principal Quantum Number',
             'Subshell', 'Atomic Orbital']
        df.set_index(index_list, inplace=True)
        return df

    def generate_ao_transition_matrix(self, electron: str = "alpha", swap_orbitals: bool = False):
        if self.pop_data is None and self.spectra_data is None:
            raise ValueError("Cannot perform an AO Projection Transition Analysis. Please check this object has both a population calculation and an excited state calculation.")
        if electron == "beta":
            ao_projection_matrix = np.copy(self.pop_data.beta_electron_data.ao_projection_matrix).real
            delta_diagonal_matrix = self.spectra_data.beta_delta_diagonal_matrix
        else:
            ao_projection_matrix = np.copy(self.pop_data.electron_data.ao_projection_matrix).real
            delta_diagonal_matrix = self.spectra_data.delta_diagonal_matrix
        if self.spectra_data.methodology == "CAS" and self.spectra_data.methodology_data.switched_orbitals is not None and swap_orbitals:
            ao_projection_matrix = parse_matrices.swap_ao_projection_orbitals(ao_projection_matrix, self.spectra_data.methodology_data.switched_orbitals)
        ao_projection_matrix = ao_projection_matrix[:, self.spectra_data.active_space_start: self.spectra_data.active_space_end]
        ao_projection_matrix = parse_matrices.convert_ao_projection_to_mo_transition(self.spectra_data.n_excitation, ao_projection_matrix)
        delta_diagonal_matrix = parse_matrices.convert_mo_transition_to_ao_projection(self.basic_data.n_mo, self.spectra_data.n_excitation, delta_diagonal_matrix)
        return np.multiply(ao_projection_matrix, delta_diagonal_matrix)
