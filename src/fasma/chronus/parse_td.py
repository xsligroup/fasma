from fasma.core.dataclasses.data import excitation
from fasma.core import messages as msg
from fasma.core import parse_excitation
from fasma.gaussian import parse_functions
from fasma.gaussian import parse_matrices
import numpy as np


def check_td(basic, file_keyword_trie, file_lines) -> bool:
    """
    Checks if .log contains a TD calculation.
    If yes, initializes and returns a CASData object.
    :param file_keyword_trie: the KeyWordTrie object of the current file
    :param file_lines: all the lines of this current file
    :param basic: the BasicData object for this current file
    :return: TDData object if .log file contains a TD calculation, none otherwise
    """
    if file_keyword_trie.find("RESP") is not None:
        try:
            full_status = file_lines[file_keyword_trie.find("DOFULL")[0] - 1].split()[2]
        except ValueError:
            full_status = "TRUE"
            pass
        if full_status == "TRUE":
            if basic.scf_type != "GHF":
                n_excited_state = int(
                    basic.n_alpha_electron * (basic.n_basis - basic.n_alpha_electron) + basic.n_beta_electron * (basic.n_basis - basic.n_beta_electron))
            else:
                n_excited_state = int(basic.homo * (basic.n_basis * 2 - basic.homo))
        elif full_status == "FALSE":
            try:
                n_excited_state = int(file_lines[file_keyword_trie.find("NROOTS")[0] - 1].split()[2])
            except ValueError:
                n_excited_state = 3
                pass
        n_active_space_mo = basic.n_mo
        n_active_space_electron = basic.n_electron
        excitation_data = excitation.TDData(n_excited_state=n_excited_state, n_active_space_mo=n_active_space_mo,
                                        n_active_space_electron=n_active_space_electron)
        ground_state_list, excited_state_list, delta_energy_list, oscillations, alpha_delta_diagonal_list, beta_delta_diagonal_list = get_excitations_td(basic, file_keyword_trie, file_lines, excitation_data)
        excitation_matrix, delta_diagonal_matrix = parse_excitation.get_excitation_matrix(ground_state_list,
                                                                                          excited_state_list,
                                                                                          delta_energy_list,
                                                                                          oscillations,
                                                                                          alpha_delta_diagonal_list)
        excitation_data.add_excitation_matrix(excitation_matrix)
        excitation_data.add_delta_diagonal_matrix(delta_diagonal_matrix)
        if basic.scf_type == "UHF":
            beta_delta_diagonal_matrix = np.vstack(beta_delta_diagonal_list)
            excitation_data.add_beta_delta_diagonal_matrix(beta_delta_diagonal_matrix)
        return excitation_data


def get_excitations_td(basic, file_keyword_trie, file_lines, excitation_data):
    ground_state_list, excited_state_list, delta_energy_list, oscillations = parse_excitation.initialize_excitation_fields(
        excitation_data.n_excitation)
    alpha_delta_diagonal_list = np.empty(excitation_data.n_excitation, dtype=np.ndarray)
    beta_delta_diagonal_list = np.empty(excitation_data.n_excitation, dtype=np.ndarray)
    num_of_results = 0
    state_lines = verify_td_completeness(file_keyword_trie, excitation_data.n_excitation)
    for x in range(excitation_data.n_ground_state):
        current_degenerate_state = x + 1
        for y in range(excitation_data.final_state - 1 - x):
            current_excited_state = x + y + 2
            ground_state_list[num_of_results] = current_degenerate_state
            excited_state_list[num_of_results] = current_excited_state

            alpha_delta_diagonal_list[num_of_results], beta_delta_diagonal_list[num_of_results], delta_energy_list[
                num_of_results], oscillations[
                num_of_results] = td_get_excited_state(basic, file_lines, state_lines[num_of_results],
                                                       excitation_data.n_active_space_mo)
            num_of_results += 1
    return ground_state_list, excited_state_list, delta_energy_list, oscillations, alpha_delta_diagonal_list, beta_delta_diagonal_list


def verify_td_completeness(file_keyword_trie, n_excitation):
    state_lines = file_keyword_trie.find("Root")
    if len(state_lines) != n_excitation:
        raise ValueError(
            str(n_excitation) + " excited states" + msg.gaussian_missing_msg())
    return state_lines


def td_get_excited_state(basic, file_lines, line_num, n_active_space_mo):
    current_line = file_lines[line_num - 1].split()
    if basic.scf_type == "GHF":
        imag_arg = {"dtype": complex}
    else:
        imag_arg = {}
    alpha_delta_diagonal = np.zeros(n_active_space_mo, **imag_arg)
    beta_delta_diagonal = np.zeros(n_active_space_mo, **imag_arg)
    energy_value = float(current_line[current_line.index("W(eV)") + 2])
    line_num += 1
    current_line = file_lines[line_num - 1].split()
    osc_value = float(current_line[current_line.index("f") + 2])
    line_num += 2

    while "->" in file_lines[line_num - 1] or "<-" in file_lines[line_num - 1]:
        current_line = file_lines[line_num - 1].replace("<-", "->").replace(">", "> ").split()
        if basic.scf_type == "GHF":
            transfer_value = complex(float(current_line[3]), float(current_line[4]))
        else:
            transfer_value = float(current_line[3])
        if "B" in current_line[0]:
            from_mo = int(current_line[0].replace("B", "")) - 1
            to_mo = int(current_line[2].replace("B", "")) - 1
            current_delta_diag = beta_delta_diagonal
        else:
            from_mo = int(current_line[0].replace("A", "")) - 1
            to_mo = int(current_line[2].replace("A", "")) - 1
            current_delta_diag = alpha_delta_diagonal
        if basic.scf_type in ["RHF", "ROHF"]:
            multiplier = 2
        elif basic.scf_type in ["UHF", "GHF"]:
            multiplier = 1
        current_delta_diag[from_mo] -= multiplier * np.dot(transfer_value, np.conjugate(transfer_value))
        current_delta_diag[to_mo] += multiplier * np.dot(transfer_value, np.conjugate(transfer_value))
        line_num += 1
    return alpha_delta_diagonal, beta_delta_diagonal, energy_value, osc_value

