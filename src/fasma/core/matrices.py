import numpy as np


def square_trig_matrix(current_matrix: np.array) -> np.array:
    """
    Returns numpy array containing a square matrix transformed from a given triangular matrix.
    :param current_matrix: the given triangular matrix to be converted
    :return: a numpy array containing a square matrix transformed from a given triangular matrix
    """
    triangular_matrix = current_matrix
    output = triangular_matrix.transpose() + triangular_matrix
    np.fill_diagonal(output, np.diag(triangular_matrix))
    return output


def combine_complex_matrix(real_matrix, imaginary_matrix):
    return real_matrix + 1j * imaginary_matrix


def convert_ao_projection_to_mo_transition(n_excitation, data):
    return np.tile(data, (n_excitation, 1))


def convert_mo_transition_to_ao_projection(n_mo, n_excitation, data):
    return np.tile(data, (1, n_mo)).reshape((n_excitation * n_mo, data.shape[1]))


def summarize_matrix(matrix):
    particle_diagonal_matrix = np.where(matrix[:, :] > 0, matrix[:, :], 0)
    hole_diagonal_matrix = np.where(matrix[:, :] < 0, matrix[:, :], 0)

    summary_matrix = np.ndarray((matrix.shape[0], 3))
    summary_matrix[:, 0] = np.sum(matrix, axis=1)
    summary_matrix[:, 1] = np.sum(particle_diagonal_matrix, axis=1)
    summary_matrix[:, 2] = np.sum(hole_diagonal_matrix, axis=1)

    return summary_matrix


def swap_ao_projection_orbitals(ao_projection_matrix, swapped_orbitals):
    swapped_matrix = ao_projection_matrix
    for i in range(swapped_orbitals.shape[0]):
        col_1 = swapped_orbitals[i, 0]
        col_2 = swapped_orbitals[i, 1]
        swapped_matrix[:, [col_1, col_2]] = swapped_matrix[:, [col_2, col_1]]
    return swapped_matrix


def interweave_matrix(a_matrix, b_matrix):
    c = np.empty((a_matrix.shape[0] + b_matrix.shape[0], a_matrix.shape[1]), dtype=a_matrix.dtype)
    c[0::2, :] = a_matrix
    c[1::2, :] = b_matrix
    return c


def calculate_ao_projection(overlap_matrix, electron_data):
    ao_projection_matrix = np.multiply(np.dot(overlap_matrix, electron_data.mo_coefficient_matrix), np.conjugate(electron_data.mo_coefficient_matrix))
    electron_data.add_ao_projection_matrix(ao_projection_matrix)