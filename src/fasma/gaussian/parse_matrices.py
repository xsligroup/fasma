import numpy as np
import math as m


def parse_mo_coefficient_matrix(basic, file_lines, start: int, last_string="S", space_skip=6, n_col=5, block_skip=3):
    n_row_block = basic.n_ao
    if basic.scf_type == "GHF":
        n_row_block *= 2
    matrix = np.zeros((n_row_block, basic.n_mo))
    cycles = m.ceil(basic.n_mo / n_col)
    n_col_block = n_col
    n_last_block_col = basic.n_mo % n_col
    skip_amount = file_lines[start - 1].rfind(last_string) + space_skip

    for current_block in range(cycles):  # Number of separated matrix blocks
        if current_block == cycles - 1 and n_last_block_col != 0:
            n_col_block = n_last_block_col
        end = start + n_row_block
        current_block_matrix = parse_matrix_block(file_lines, start, n_row_block, n_col_block, skip_amount)
        matrix[:, (n_col * current_block): (n_col * current_block) + n_col_block] = current_block_matrix
        start = end + block_skip
    return matrix


def parse_matrix_line(current_line, skip_amount) -> np.array:
    line = current_line[skip_amount:].replace("**********", "   nan")
    if "*" in line:
        line = line.replace("*", " ")
    if "D" in current_line:
        line = line.replace("D", "E").split()
    else:
        line = line.replace("-", " -").split()
    try:
        value_array = np.asarray(line, dtype=float)
    except ValueError:
        print(line)
        print(current_line)
    return value_array


def parse_matrix_block(file_lines, start, n_row, n_col, skip_amount):
    block_matrix = np.zeros((n_row, n_col))
    for current_row in range(n_row):  # Number of rows
        line = file_lines[start - 1 + current_row]
        line_values = parse_matrix_line(line, skip_amount)
        try:
            block_matrix[current_row, : len(line_values)] = line_values
        except ValueError:
            print(line)
            print(line_values)
            print(start - 1 + current_row)
    return block_matrix


def parse_matrix(file_lines, start: int, n_row, last_string="1", space_skip=2, n_col=5, block_skip=1, triangular=False):
    # Parse cartesian_axes, last_string="s", space_skip=6
    matrix = np.zeros((n_row, n_row))
    cycles = m.ceil(n_row / n_col)
    n_row_block = n_row
    n_col_block = n_col
    n_last_block_col = n_row % n_col
    if last_string == "S":
        skip_amount = file_lines[start - 1].rfind(last_string) + space_skip
    else:
        skip_amount = file_lines[start - 1].find(last_string) + space_skip

    for current_block in range(cycles):  # Number of separated matrix blocks
        if triangular:
            n_row_block = n_row - (current_block * n_col)
        if current_block == cycles - 1 and n_last_block_col != 0:
            n_col_block = n_last_block_col
        end = start + n_row_block
        current_block_matrix = parse_matrix_block(file_lines, start, n_row_block, n_col_block, skip_amount)
        matrix[n_row - n_row_block:, (n_col * current_block): (n_col * current_block) + n_col_block] = current_block_matrix
        start = end + block_skip
    return matrix
