from fasma.core.dataclasses.data import basic


def get_basic(file_keyword_trie, file_lines):
    """
    Initializes a BasicData object containing basic information about this .log calculation.
    :param file_keyword_trie: the KeyWordTrie object of the current file
    :param file_lines: all the lines of this current file
    :return:
    """
    atom_list = get_atom_list(file_keyword_trie, file_lines)
    scf_type = find_scf_type(file_keyword_trie, file_lines)
    temp = find_basic(file_keyword_trie, file_lines)

    n_basis = temp[0]
    n_primitive_gaussian = temp[1]
    n_electron = temp[2]

    if scf_type == "GHF":
        n_alpha_electron = n_electron // 2
    else:
        n_alpha_electron = count_electrons_type(file_keyword_trie, file_lines, "alpha")
    if scf_type == "RHF" or scf_type == "GHF":
        n_beta_electron = n_alpha_electron
    else:
        n_beta_electron = count_electrons_type(file_keyword_trie, file_lines, "beta")

    n_mo = n_basis
    if scf_type == "GHF":
        n_mo *= 2
    homo = n_electron
    if scf_type == "ROHF" or scf_type == "RHF":
        homo = homo // 2
    lumo = homo + 1

    basic_data = basic.BasicData(atom_list=atom_list, scf_type=scf_type, n_basis=n_basis,
                                 n_primitive_gaussian=n_primitive_gaussian, n_alpha_electron=n_alpha_electron,
                                 n_beta_electron=n_beta_electron, n_electron=n_electron, n_mo=n_mo, homo=homo,
                                 lumo=lumo)
    return basic_data


def get_atom_list(file_keyword_trie, file_lines) -> list:
    """
    Finds all the atoms present in the molecule of the .out file and returns them as a list.
    :return: the list of all atoms in the molecule of the .out file
    """
    current_line_num = file_keyword_trie.find("GEOM:")[0] + 1

    atom_list = []

    while file_lines[current_line_num - 1].strip():
        current_line = file_lines[current_line_num - 1].split()
        atom_list.append(current_line[0].upper())
        current_line_num += 1

    return atom_list


def find_scf_type(file_keyword_trie, file_lines) -> str:
    """
    Find and return the SCF type of the .out file.
    :return: a string containing the SCF type of the given .out file
    """

    line = file_lines[file_keyword_trie.find("REFERENCE")[0] - 1]

    if "COMPLEX" in line or "REAL" in line:
        scf_position = 3
    else:
        scf_position = 2
    type_indicator = line.split()[scf_position][0]
    if type_indicator == "R" and line.split()[scf_position][1] == "O":
        type_indicator = "RO"
    type_dict = {"R": "RHF", "RO": "ROHF", "U": "UHF", "G": "GHF", "X": "GHF"}
    return type_dict.get(type_indicator)


def find_basic(file_keyword_trie, file_lines) -> list:
    """
    Find and parse basic attributes of a .out file and return them in a list.
    :return: a list containing the basic attributes in each .out file.
    """
    word_conveyor = ["NBasis", "NPrimitive", "Total Electrons"]
    variable_conveyor = []

    for x in word_conveyor:
        current_line = file_lines[file_keyword_trie.find(x)[0] - 1].split(x)
        variable_conveyor.append(int(current_line[1]))
    return variable_conveyor


def count_electrons_type(file_keyword_trie, file_lines, e_type: str = "alpha") -> int:
    electron_count = 0
    if e_type == "beta":
        keyword = "Eigenenergies (Beta)"
    else:
        keyword = "Eigenenergies (Alpha)"
    current_line_num = file_keyword_trie.find(keyword)[0] + 3
    while file_lines[current_line_num - 1].strip():
        current_line = file_lines[current_line_num - 1].split()
        electron_count += len(current_line)
        current_line_num += 1
    return electron_count
