"""Various tools to working with binary linear codes."""

from blincodes import matrix, vector


def make_generator(mat):
    """Return the generator matrix from general matrix `mat`."""
    return matrix.Matrix(
        (row.value for row in mat.diagonal_form if row.value),
        mat.ncolumns)


def make_parity_check(mat):
    """Return the parity-check matrix from generator matrix `mat`."""
    return mat.orthogonal


def hadamard_product(generator_a, generator_b):
    """Evaluate the generator matrix of Hadamard product code.

    :param: Matrix generator_a -  the generator matrix of the first code;
    :param: Matrix generator_b -  the generator matrix of the second code.
    :return: Matrix generator - the generator matrix of Hadamard product of
                                the first and the second codes.
    """
    hadamard_dict = {}  # {index of the fist 1 in the row: row}
    hadamard = []
    for row_a in generator_a:
        for row_b in generator_b:
            row = row_a * row_b
            test_row = row.copy()
            for i, row_h in hadamard_dict.items():
                if test_row[i]:
                    test_row += row_h
            if test_row.value:
                hadamard_dict[test_row.support[0]] = test_row
                hadamard.append(row)
    return matrix.from_vectors(hadamard)


def intersection(generator_a, generator_b):
    """Return generator matrix of intersection of two codes."""
    return make_parity_check(matrix.concatenate(
        make_parity_check(generator_a),
        make_parity_check(generator_b), by_rows=True))


def union(generator_a, generator_b):
    """Return generator matrix of union of two codes."""
    return make_generator(matrix.concatenate(
        generator_a, generator_b, by_rows=True))


def puncture(generator, columns=None, remove_zeroes=False):
    """Return generator matrix of punctured code.

    Punctured code is code obtaining by set the positions
    with indexes from `ncolumns` of every codeword to zero.

    Punctured code is NOT subcode of original code!
    """
    if not columns:
        columns = []
    mask = vector.from_support_supplement(generator.ncolumns, columns)
    puncture_matrix = matrix.Matrix(
        ((row * mask).value for row in generator),
        generator.ncolumns).diagonal_form
    if remove_zeroes:
        return matrix.Matrix(
            (row.value for row in puncture_matrix if row.value),
            generator.ncolumns).submatrix(
                columns=(i for i in range(generator.ncolumns)
                         if i not in columns))
    return matrix.Matrix(
        (row.value for row in puncture_matrix if row.value),
        generator.ncolumns)


def truncate(generator, columns=None, remove_zeroes=False):
    """Return generator matrix of truncated code.

    Truncated code is code obtaining by choose codewords which
    have coordinates with indexes from `columns` is zero.

    Unlike the punctured code truncated code is a subcode of original code.

    NOTE! If remove_zeroes is set to True the truncated codes would not be
    a subcode of the original code.
    """
    if not columns:
        columns = []
    mask = vector.from_support(generator.ncolumns, columns)
    trunc = matrix.Matrix(
        (row.value for row in generator.gaussian_elimination(columns)
         if not (row * mask).value),
        generator.ncolumns).echelon_form
    trunc = matrix.Matrix((row.value for row in trunc if row.value),
                          generator.ncolumns)
    if remove_zeroes:
        return trunc.submatrix(
            columns=(i for i in range(generator.ncolumns)
                     if i not in columns))
    return trunc


def hull(generator):
    """Evaluate the generator matrix of the code's hull.

    The code's hull is intersection of code and it's dual.
    """
    return make_parity_check(
        matrix.concatenate(generator,
                           make_parity_check(generator),
                           by_rows=True))


def iter_codewords(generator):
    """Iterate over all codewords of code."""
    for i in range(1 << generator.nrows):
        yield (matrix.Matrix([i], generator.nrows) * generator)[0]


def spectrum(generator):
    """Return the spectrum of code."""
    spec = {i: 0 for i in range(generator.ncolumns + 1)}
    for vec in iter_codewords(generator):
        spec[vec.hamming_weight] += 1
    return spec


def encode(generator, vec):
    """Encode the `vec` using generator matrix `generator` of code."""
    try:
        return (matrix.from_vectors([vec]) * generator)[0]
    except TypeError:
        pass
    except IndexError:
        return None
    try:
        return (vec * generator)[0]
    except IndexError:
        pass
    return None


def syndrome(parity_check, vec):
    """Return the syndrome of `vec` using parity check matrix."""
    try:
        return (parity_check * matrix.from_vectors([vec]).T).T[0]
    except TypeError:
        pass
    except IndexError:
        return None
    try:
        return (parity_check * vec.T).T[0]
    except IndexError:
        pass
    return None
