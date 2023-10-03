import numpy as np
import math as m


def parse_mo_coefficient_matrix(basic, file_lines, start: int, last_string="S", space_skip=6, n_col=4, block_skip=4):
    n_row_block = basic.n_basis
    matrix = np.zeros((n_row_block, basic.n_mo))
    cycles = m.ceil(basic.n_mo / n_col)
    n_col_block = n_col
    n_last_block_col = basic.n_mo % n_col
    skip_amount = file_lines[start - 1].rfind(last_string) + space_skip

    for current_block in range(cycles):  # Number of separated matrix blocks
        if current_block == cycles - 1 and n_last_block_col != 0:
            n_col_block = n_last_block_col
        end = start + n_row_block + len(basic.atom_list)
        current_block_matrix = parse_matrix_block(file_lines, start, n_row_block, n_col_block, skip_amount)
        matrix[:, (n_col * current_block): (n_col * current_block) + n_col_block] = current_block_matrix
        start = end + block_skip
    return matrix


def parse_matrix_line(current_line, skip_amount) -> np.array:
    line = current_line[skip_amount:].replace("**********", "   nan").split()
    value_array = np.asarray(line, dtype=float)
    return value_array


def parse_matrix_block(file_lines, start, n_row, n_col, skip_amount):
    block_matrix = np.zeros((n_row, n_col))
    empty_space = 0
    for current_row in range(n_row):  # Number of rows
        line = file_lines[start - 1 + current_row + empty_space]
        while not line.strip():
            empty_space += 1
            line = file_lines[start - 1 + current_row + empty_space]
        line_values = parse_matrix_line(line, skip_amount)
        block_matrix[current_row, : len(line_values)] = line_values
    return block_matrix

