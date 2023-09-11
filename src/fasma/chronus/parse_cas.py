from fasma.core.dataclasses.data import methodology, excitation
from fasma.core import messages as msg
from fasma.core import parse_excitation
import numpy as np


def check_cas(basic, file_keyword_trie, file_lines) -> bool:
    """
    Checks if .log contains a CAS calculation.
    If yes, initializes and returns a CASData object.
    :param file_keyword_trie: the KeyWordTrie object of the current file
    :param file_lines: all the lines of this current file
    :param basic: the BasicData object for this current file
    :return: CASData object if this .log file contains a CAS calculation, none otherwise
    """
    if file_keyword_trie.find("MCSCF") is not None or int(file_lines[file_keyword_trie.find("OSCISTREN")[0] - 1].split()[2]) >= 1:
        n_active_space_electron =  int(file_lines[file_keyword_trie.find("NACTE")[0] - 1].split()[2])
        n_active_space_mo = int(file_lines[file_keyword_trie.find("NACTO")[0] - 1].split()[2])
        n_root = int(file_lines[file_keyword_trie.find("NROOTS")[0] - 1].split()[2])
        n_ground_state = int(file_lines[file_keyword_trie.find("OSCISTREN")[0] - 1].split()[2])
        final_state_full = n_root
        active_space_start = basic.homo - n_active_space_electron
        final_state = n_root
        n_excitation_full = int((final_state_full * n_ground_state) - (n_ground_state * (n_ground_state + 1) / 2))

        cas_data = methodology.CASCentricData(n_root=n_root, n_excitation_full=n_excitation_full)
        excitation_data = excitation.CASData(n_ground_state=n_ground_state, final_state=final_state,
                                     n_active_space_mo=n_active_space_mo,
                                     n_active_space_electron=n_active_space_electron,
                                     active_space_start=active_space_start, methodology_data=cas_data)
        excitation_matrix, delta_diagonal_matrix = parse_excitation.get_excitation_matrix(*get_excitations_cas(file_keyword_trie, file_lines, excitation_data))
        excitation_data.add_excitation_matrix(excitation_matrix)
        excitation_data.add_delta_diagonal_matrix(delta_diagonal_matrix)
        return excitation_data


def get_excitations_cas(file_keyword_trie, file_lines, excitation_data):
    ground_state_list, excited_state_list, delta_energy_list, oscillations, delta_diagonal_list = parse_excitation.initialize_excitation_fields(
        excitation_data.n_excitation)
    num_of_results = 0
    diag_lines, excitation_lines = verify_cas_completeness(file_keyword_trie, excitation_data.methodology_data.n_root,
                                                                  excitation_data.methodology_data.n_excitation_full)
    for x in range(excitation_data.n_ground_state):
        current_degenerate_state = x + 1
        current_degenerate_diag = dm_get_diag(file_lines, diag_lines[x], excitation_data.n_active_space_mo)
        for y in range(excitation_data.final_state - 1 - x):
            current_excited_state = x + y + 2
            ground_state_list[num_of_results] = current_degenerate_state
            excited_state_list[num_of_results] = current_excited_state
            delta_diagonal_list[num_of_results], delta_energy_list[num_of_results], oscillations[
                num_of_results] = cas_get_excited_state(file_lines, current_degenerate_diag,
                                                        diag_lines[current_excited_state - 1],
                                                        excitation_lines[num_of_results],
                                                        excitation_data.n_active_space_mo)
            num_of_results += 1
    return ground_state_list, excited_state_list, delta_energy_list, oscillations, delta_diagonal_list


def verify_cas_completeness(file_keyword_trie, n_root, n_excitation):
    diag_lines = file_keyword_trie.find("State " + str(n_root) + ":")
    if len(diag_lines) != 1:
        raise ValueError(
            str(n_root) + " PDM diagonals" + msg.gaussian_missing_msg())
    start = file_keyword_trie.find("State 1:")[0]
    diag_n_lines = file_keyword_trie.find("State 2:")[0] - start
    diag_lines = [start + x * diag_n_lines for x in range(0, n_root)]

    excitation_lines = file_keyword_trie.find("E(Eh)")
    if len(excitation_lines) != n_excitation:
        raise ValueError(str(n_excitation) + " excitations" + msg.gaussian_missing_msg())
    return diag_lines, excitation_lines


def cas_get_excited_state(file_lines, ground_diag, diag_line_num, excitation_line_num, n_active_space_mo):
    delta_diagonal = dm_get_diag(file_lines, diag_line_num, n_active_space_mo) - ground_diag
    energy_value, osc_value = dm_get_excitation(file_lines, excitation_line_num)
    return delta_diagonal, energy_value, osc_value


def dm_get_excitation(file_lines, line_num):
    current_line = file_lines[line_num - 1].split()
    return (float(current_line[8]) * 27.2114), float(current_line[11])


def dm_get_diag(file_lines, line_num, n_active_space_mo):
    diag = np.zeros(n_active_space_mo, dtype=float)
    counter = 0
    while file_lines[line_num - 1].strip():
        stripped_line = file_lines[line_num - 1].replace("(", " ").replace(")", "").split()[3::2]
        diag[counter: counter + len(stripped_line)] = np.asarray(stripped_line, dtype=float)
        counter += len(stripped_line)
    return diag
