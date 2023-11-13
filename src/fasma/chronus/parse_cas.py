from fasma.core.dataclasses.data import methodology, excitation
from fasma.core import messages as msg
from fasma.core import parse_excitation
from fasma.core import conversion
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
    if file_keyword_trie.find("MCSCF") is not None and int(file_lines[file_keyword_trie.find("OSCISTREN")[0] - 1].split()[2]) >= 1:
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
    ground_state_list, excited_state_list, delta_energy_list, oscillations = parse_excitation.initialize_excitation_fields(
        excitation_data.n_excitation)
    delta_diagonal_list = np.empty(excitation_data.n_excitation, dtype=np.ndarray)
    num_of_results = 0
    diag_lines, excitation_lines, diag_n_lines = verify_cas_completeness(file_keyword_trie, excitation_data.methodology_data.n_root,
                                                                  excitation_data.methodology_data.n_excitation_full)
    state_diag_matrix = get_state_diag_matrix(file_lines, diag_lines, excitation_data.n_active_space_mo,
                                              diag_n_lines)
    for x in range(excitation_data.n_ground_state):
        current_degenerate_state = x + 1
        current_degenerate_diag = state_diag_matrix[x]
        for y in range(excitation_data.final_state - 1 - x):
            current_excited_state = x + y + 2
            ground_state_list[num_of_results] = current_degenerate_state
            excited_state_list[num_of_results] = current_excited_state
            delta_diagonal_list[num_of_results], delta_energy_list[num_of_results], oscillations[
                num_of_results] = cas_get_excited_state(file_lines, current_degenerate_diag,
                                                        state_diag_matrix[current_excited_state - 1],
                                                        excitation_lines[num_of_results])
            num_of_results += 1
    return ground_state_list, excited_state_list, delta_energy_list, oscillations, delta_diagonal_list


def get_state_diag_matrix(file_lines, diag_lines, n_active_space_mo, diag_n_lines):
    state_diag_matrix = np.zeros((len(diag_lines), n_active_space_mo), dtype=float)
    for x in range(len(diag_lines)):
        state_diag_matrix[x, :] = dm_get_diag(file_lines, diag_lines[x], n_active_space_mo, diag_n_lines)
    return state_diag_matrix


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
    return diag_lines, excitation_lines, diag_n_lines


def cas_get_excited_state(file_lines, ground_diag, excited_diag, excitation_line_num):
    delta_diagonal = excited_diag - ground_diag
    energy_value, osc_value = dm_get_excitation(file_lines, excitation_line_num)
    return delta_diagonal, energy_value, osc_value


def dm_get_excitation(file_lines, line_num):
    current_line = file_lines[line_num - 1].split()
    return (float(current_line[8]) * conversion.EV), float(current_line[11])


def dm_get_diag(file_lines, line_num, n_active_space_mo, diag_n_lines):
    diag = np.zeros(n_active_space_mo, dtype=float)
    counter = 0
    for x in range(diag_n_lines):
        stripped_line = file_lines[line_num - 1].replace("(", " ").replace(")", "").split()
        if "State" in stripped_line:
            start = 3
        else:
            start = 1
        stripped_line = stripped_line[start::2]
        diag[counter: counter + len(stripped_line)] = np.asarray(stripped_line, dtype=float)
        counter += len(stripped_line)
        line_num += 1
    return diag
