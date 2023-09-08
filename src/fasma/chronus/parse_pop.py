from fasma.core.dataclasses.data import pop, electron
from fasma.chronus import parse_matrices
from fasma.core import matrices
from fasma.core import messages as msg
import numpy as np
import math as m


def check_pop(basic, bin_file, file_keyword_trie, file_lines) -> bool:
    """
    Tests if .log contains a pop calculation.
    If yes, initializes necessary pop-related attributes with SCF type in consideration.
    :return: true if this .log contains a population calculation, false otherwise
    """
    try:
        temp_check = int(file_lines[file_keyword_trie.find("PRINTMOS")[0] - 1].split()[2])
    except TypeError:
        pass
    else:
        if temp_check in [1, 3, 5, 7, 9]:
            try:
                neo_status = file_lines[file_keyword_trie("NEO") - 1].split()[2]
            except TypeError:
                neo_status = False
            else:
                neo_status = neo_status == "True"
            ao_matrix = get_ao_matrix(basic, file_keyword_trie, file_lines)
            if bin_file is None:
                #overlap_matrix = get_overlap_matrix(basic, file_keyword_trie, file_lines)
                print("Overlap Matrix isn't printed in .out file yet. Please pass in an bin_file in addition to .out file")
            else:
                overlap_matrix = bin_file["/INTS/OVERLAP"][:, :]
            if basic.scf_type == "GHF":
                identity = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.complex128)
                overlap_matrix = np.kron(overlap_matrix, identity)
            electron_data = get_alpha_electron_data(basic, bin_file, file_keyword_trie, file_lines, neo_status)
            matrices.calculate_ao_projection(overlap_matrix, electron_data)
            pop_data = pop.PopData(ao_matrix=ao_matrix, overlap_matrix=overlap_matrix, electron_data=electron_data)
            if basic.scf_type == "UHF":
                beta_electron_data = get_beta_electron_data(basic, bin_file, file_keyword_trie, file_lines, neo_status)
                matrices.calculate_ao_projection(overlap_matrix, beta_electron_data)
                pop_data.add_beta_electron_data(beta_electron_data)
            return pop_data
    return


def get_ao_matrix(basic, file_keyword_trie, file_lines):
    start = file_keyword_trie.find("EigV")[0] + 2
    ao_matrix = np.empty((basic.n_mo, 5), dtype='<U12')
    subshell_position = file_lines[start - 1].rfind('S')
    counter = 0
    n_lines = basic.n_basis + len(basic.atom_list)
    for current_ao in range(n_lines):
        line = file_lines[start - 1 + current_ao]
        if line.strip():
            ao_row = parse_ao_line(line, subshell_position)
            if len(ao_row) == 5:
                atom_info = ao_row[0:2]
            else:
                ao_row = atom_info + ao_row
            ao_row = np.array(ao_row, dtype='<U12')
            ao_matrix[counter, :] = ao_row
            counter += 1
            if basic.scf_type == "GHF":
                ao_matrix[counter, :] = ao_row
                counter += 1
    return ao_matrix


def parse_ao_line(current_line, subshell_position):
    subshell = current_line[subshell_position]
    if subshell == "S":
        atomic_orbital = "0"
    else:
        atomic_orbital = current_line[subshell_position + 1: subshell_position + 3].strip()
    line = current_line[0:subshell_position] + " " + subshell + " " + atomic_orbital
    line = line.split()[1:]
    line[1] = line[1].split("-")[0]
    return line


def get_alpha_electron_data(basic, bin_file, file_keyword_trie, file_lines, neo_status):
    #density_matrix = get_density_matrix(basic=basic, file_keyword_trie=file_keyword_trie, file_lines=file_lines, cas_status=cas_status)
    if bin_file is None:
        mo_coefficient_matrix = get_mo_coefficient_matrix(basic=basic, file_keyword_trie=file_keyword_trie, file_lines=file_lines, neo_status=neo_status)
    else:
        mo_coefficient_matrix = bin_file["/SCF/MO1"][:, :].transpose()
    if basic.scf_type == "GHF":
        if bin_file is None:
            beta_mo_coefficient_matrix = get_mo_coefficient_matrix(basic=basic, file_keyword_trie=file_keyword_trie,
                                      file_lines=file_lines, beta=True, neo_status=neo_status)
        else:
            beta_mo_coefficient_matrix = mo_coefficient_matrix[basic.n_basis:, :]
            mo_coefficient_matrix = mo_coefficient_matrix[:basic.n_basis, :]
        mo_coefficient_matrix = matrices.interweave_matrix(mo_coefficient_matrix, beta_mo_coefficient_matrix)
    eigenvalues = get_eigenvalues(basic=basic, file_keyword_trie=file_keyword_trie, file_lines=file_lines, n_mo=basic.n_mo, neo_status=neo_status)
    return electron.ElectronData(mo_coefficient_matrix, eigenvalues)


def get_beta_electron_data(basic, bin_file, file_keyword_trie, file_lines, neo_status):
    #density_matrix = get_density_matrix(basic=basic, file_keyword_trie=file_keyword_trie, file_lines=file_lines, cas_status=cas_status, beta=True)
    #beta_electron_data = electron.BetaData(density_matrix=density_matrix)
    if basic.scf_type == "UHF":
        if bin_file is None:
            get_mo_coefficient_matrix(basic=basic, file_keyword_trie=file_keyword_trie, file_lines=file_lines, beta=True, neo_status=neo_status)
        else:
            mo_coefficient_matrix = bin_file["/SCF/MO2"][:, :].transpose()
        eigenvalues = get_eigenvalues(basic=basic, file_keyword_trie=file_keyword_trie, file_lines=file_lines, n_mo=basic.n_mo, beta=True, neo_status=neo_status)
    beta_electron_data = electron.BetaData(mo_coefficient_matrix=mo_coefficient_matrix, eigenvalues=eigenvalues)
    return beta_electron_data


def get_mo_coefficient_matrix(basic, file_keyword_trie, file_lines, n_col=4, neo_status=False, beta=False):
    cycles = m.ceil(basic.n_mo / n_col)
    neo_dict = {"RHF": 1, "ROHF": 2, "UHF": 2, "GHF": 4}
    if neo_status:
        neo_status = neo_dict.get(basic.scf_type)
    if beta:
        index = cycles * (1 + (basic.scf_type == "GHF") + neo_status)
    else:
        index = 0 + (cycles * neo_status)
    start = file_keyword_trie.find("EigV")[index] + 2
    mo_coefficient_matrix = parse_matrices.parse_mo_coefficient_matrix(basic, file_lines, start=start)
    if basic.scf_type == "GHF":
        start = file_keyword_trie.find("EigV")[index + cycles] + 2
        imaginary_matrix = parse_matrices.parse_mo_coefficient_matrix(basic, file_lines, start=start)
        mo_coefficient_matrix = matrices.combine_complex_matrix(mo_coefficient_matrix, imaginary_matrix)
    return mo_coefficient_matrix


def get_eigenvalues(basic, file_keyword_trie, file_lines, n_mo, n_col=4, neo_status=False, beta=False):
    cycles = m.ceil(n_mo / n_col)
    neo_dict = {"RHF": 1, "ROHF": 2, "UHF": 2, "GHF": 4}
    if neo_status:
        neo_status = neo_dict.get(basic.scf_type)
    if beta:
        index = cycles * (1 + (basic.scf_type == "GHF") + neo_status)
    else:
        index = 0 + (neo_status * cycles)
    n_col_block = n_col
    n_last_block_col = n_mo % n_col

    eigenvalue_type = "alpha"
    if beta:
        eigenvalue_type = "beta"
    eigenvalue_lines = file_keyword_trie.find("EigV")[index: index + cycles]

    if len(eigenvalue_lines) != cycles:
        raise ValueError(str(n_mo) + " " + eigenvalue_type + " eigenvalues" + msg.gaussian_missing_msg())

    skip_amount = file_lines[eigenvalue_lines[0] - 1].find("-") + 2
    eigenvalues = np.zeros(n_mo, dtype=float)

    for current_block in range(cycles):
        if current_block == cycles - 1 and n_last_block_col != 0:
            n_col_block = n_last_block_col
        line = file_lines[eigenvalue_lines[current_block] - 1]
        line_values = parse_matrices.parse_matrix_line(line, skip_amount)
        eigenvalues[(n_col * current_block): (n_col * current_block) + n_col_block] = line_values

    return eigenvalues
