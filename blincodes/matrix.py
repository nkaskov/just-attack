"""Module for working with matrices over GF(2) field."""

from random import randint, sample
import math
from blincodes import vector


class Matrix():
    """Binary matrix abstraction."""

    def __init__(self, value=None, ncolumns=0):
        """Create new matrix.

        :param: value - any iterable of integers
        :param: ncolumns - number of columns in the matrix
        """
        if not isinstance(ncolumns, int):
            raise TypeError(
                'expected `ncolumns` is integer, but '
                'got {}'.format(type(ncolumns)))
        if ncolumns < 0:
            raise ValueError(
                'expected `ncolumns` is not less then 0, but '
                '{} < 0'.format(ncolumns))
        self._ncolumns = ncolumns
        if not value:
            value = []
        if self._ncolumns:
            self._matrix = tuple(vector.Vector(i, ncolumns) for i in value)
        else:
            self._matrix = tuple()
        if not self._matrix:
            self._ncolumns = 0

    @property
    def nrows(self):
        """Return number of rows."""
        return len(self._matrix)

    @property
    def ncolumns(self):
        """Return number of columns."""
        return self._ncolumns

    @property
    def shapes(self):
        """Return shapes of the matrix: (nrows, ncolumns)."""
        return self.nrows, self.ncolumns

    @property
    def rank(self):
        """Evaluate the rank of the matrix."""
        matrix_rows = tuple(self.copy())
        rank_value = 0
        for i, row in enumerate(matrix_rows):
            for j in range(self.ncolumns):
                if row[j]:
                    rank_value += 1
                    break
            else:
                continue
            for row2 in (v for k, v in enumerate(matrix_rows)
                         if k > i and matrix_rows[k][j]):
                row2 += row
        return rank_value

    @property
    def echelon_form(self):
        """Evaluate the echelon form of the matrix."""
        matrix_rows = tuple(self.copy())
        rank_value = 0
        for i, row in enumerate(matrix_rows):
            for j in range(self.ncolumns):
                if row[j]:
                    rank_value += 1
                    break
            else:
                continue
            for row2 in (v for k, v in enumerate(matrix_rows)
                         if k > i and matrix_rows[k][j]):
                row2 += row
        return Matrix(
            sorted((row.value for row in matrix_rows), reverse=True),
            self.ncolumns)

    @property
    def diagonal_form(self):
        """Evaluate the diagonal form of the matrix."""
        matrix_rows = tuple(self.copy())
        rank_value = 0
        for i, row in enumerate(matrix_rows):
            for j in range(self.ncolumns):
                if row[j]:
                    rank_value += 1
                    break
            else:
                continue
            for row2 in (v for k, v in enumerate(matrix_rows)
                         if k != i and matrix_rows[k][j]):
                row2 += row
        return Matrix(
            sorted((row.value for row in matrix_rows), reverse=True),
            self.ncolumns)

    @property
    def inverse(self):
        """Evaluate the inverse matrix."""
        matrix_rows = tuple(self.copy())
        identity_rows = tuple(identity(self.nrows))
        rank_value = 0
        for i, row in enumerate(zip(matrix_rows, identity_rows)):
            for j in range(self.ncolumns):
                if row[0][j]:
                    rank_value += 1
                    break
            else:
                continue
            for row2, row3 in ((v[0], v[1])
                               for k, v in enumerate(zip(matrix_rows,
                                                         identity_rows))
                               if k != i and matrix_rows[k][j]):
                row2 += row[0]
                row3 += row[1]
        return Matrix(
            (row.value for _, row in sorted(zip(matrix_rows, identity_rows),
                                            key=lambda x: x[0].value,
                                            reverse=True)),
            self.nrows)

    @property
    def orthogonal(self):
        """Return transposed orthogonal matrix.

        Transposed orthogonal matrix H is matrix satisfied
        the following condition
              self * H^T = 0.
        Moreover the matrix H has `(ncolumns - self.rank)` rows
        and `self.ncolumns` columns.
        """
        # Evaluation of diagonal form.
        matrix_rows = tuple(self.copy())
        identity_columns = []
        for i, row in enumerate(matrix_rows):
            for j in range(self.ncolumns):
                if row[j]:
                    identity_columns.append(j)
                    break
            else:
                continue
            for row2 in (v for k, v in enumerate(matrix_rows)
                         if k != i and matrix_rows[k][j]):
                row2 += row
        matrix_rows = sorted((row for row in matrix_rows),
                             key=lambda x: x.value,
                             reverse=True)
        # Delete zero rows and represent matrix as list of strings.
        str_rows = [str(row) for row in matrix_rows if row.value]
        # Delete identity matrix
        str_rows = list(map(
            lambda x: ''.join((sym for i, sym in enumerate(x)
                               if i not in identity_columns)),
            str_rows))
        # Insert identity matrix in the right place
        # and sort the matrix according to list of identity columns
        matrix_rows = []
        counter = 0
        nrows = self.ncolumns - len(identity_columns)
        for i in range(self.ncolumns):
            if i in identity_columns:
                matrix_rows.append(str_rows.pop(0))
            else:
                matrix_rows.append(
                    '0'*counter + '1' + '0'*(nrows - counter - 1))
                counter += 1
        # Transpose the matrix and construct the Matrix object
        matrix_rows = tuple(int(''.join(el), 2) for el in zip(*matrix_rows))
        if not matrix_rows:
            return Matrix([0], self.ncolumns)
        return Matrix(matrix_rows, self.ncolumns)

    @property
    def T(self):
        """Return transpose of matrix."""
        return self.transpose()

    def to_str(self, zerofillers=None, onefillers=None, numbered=False):
        """Return string representation of matrix."""
        matrix_str = ''
        if self._matrix:
            number_formated = '{{: >{}}}: '.format(
                int(math.log10(self.nrows)) + 1)
            for i, vec in enumerate(self._matrix):
                if numbered:
                    matrix_str += number_formated.format(i)
                matrix_str += vec.to_str(zerofillers, onefillers)
                matrix_str += '\n'
        return matrix_str[:-1]

    def to_latex_str(self):
        """Return representation of matrix as LaTeX string."""
        return '\\\\\n'.join(tuple(row.to_latex_str() for row in self))

    def copy(self):
        """Make copy of the matrix."""
        return Matrix(
            (row.value for row in self),
            self.ncolumns)

    def submatrix(self, columns=None):
        """Return matrix contained in columns."""
        if not columns:
            return self
        columns = tuple(columns)
        sub_matr = []
        for row in self:
            value = 0
            for i in columns:
                value <<= 1
                if row[i]:
                    value ^= 1
            sub_matr.append(value)
        return Matrix(sub_matr, len(columns))

    def transpose(self):
        """Return transposition of matrix."""
        return Matrix(
            (int(''.join(el), 2)
             for el in zip(*(str(row) for row in self))),
            self.nrows)

    def concatenate(self, other, by_rows=False):
        """Concatenate two matrices."""
        if by_rows:
            self._ncolumns = max(self.ncolumns, other.ncolumns)
            self._matrix = tuple(
                vector.Vector(row.value, self._ncolumns)
                for row in self._matrix + tuple(other))
        else:
            self._matrix = tuple(row_self.concatenate(row_other)
                                 for row_self, row_other in zip(self, other))
            self._ncolumns = self.ncolumns + other.ncolumns
        if len(self._matrix) == 0:
            return self.__class__()
        return self

    def is_zero(self):
        """Return True if any element of matrix is zero."""
        for row in self:
            if row.value:
                return False
        return True

    def is_max_rank(self):
        """Return True if matrix has maximal rank."""
        if self.rank == min(self.nrows, self.ncolumns):
            return True
        return False

    def is_identity(self):
        """Return True if matrix is identity matrix."""
        if not self.ncolumns:
            return False
        mask = (1 << (self.ncolumns - 1))
        for row in self:
            if row.value != mask:
                return False
            mask >>= 1
        return True

    def solve(self, vect_b):
        """Solve linear equation Ax^T = vect_b^T."""
        if not vect_b.value:
            # vect_b == 0
            return (self.orthogonal, vector.Vector(0, self.ncolumns))
        extend_mat = concatenate(self, Matrix(vect_b, 1))
        orthogonal = extend_mat.orthogonal
        fundamental = Matrix(
            (row.value >> 1 for row in orthogonal if not row[-1]),
            self.ncolumns)
        vec_solve = [row for row in orthogonal if row[-1]]
        if vec_solve:
            return (fundamental, (vec_solve[0] >> 1).set_length(self.ncolumns))
        return None, None

    def gaussian_elimination(self, columns=None, sort=True):
        """Evaluate the Gaussian eliminations on columns `columns`.

        :param: `iterable` columns - list or any iterable of columns.
        :return: Gaussian eliminations evaluated on columns.
        """
        if not columns:
            columns = tuple(range(self.ncolumns))
        else:
            columns = tuple(col for col in range(self.ncolumns)
                            if col in columns)
        matrix_rows = tuple(self.copy())
        for i, row in enumerate(matrix_rows):
            for j in columns:
                if row[j]:
                    for row2 in (v for k, v in enumerate(matrix_rows)
                                 if k != i and matrix_rows[k][j]):
                        row2 += row
                    break
            else:
                continue
        if sort:
            mask = vector.from_support(self.ncolumns, support=columns).value
            return Matrix(
                sorted((row.value for row in matrix_rows),
                       key=lambda x: x & mask,
                       reverse=True),
                self.ncolumns)
        return Matrix(
            (row.value for row in matrix_rows),
            self.ncolumns)

    def __bool__(self):
        """Return True if and only if nrows > 0."""
        if self.nrows == 0:
            return False
        return True

    def __iter__(self):
        """Iterate over rows of matrix."""
        for vec in self._matrix:
            yield vec

    def __getitem__(self, index):
        """Return row of matrix with index `index`.

        If index is integer then it returns the row with index `index`.
        If index is slice the it returns the Matrix object.
        """
        if isinstance(index, int):
            return self._matrix[index]
        try:
            submatrix = tuple(self._matrix[i].value
                              for i in range(*index.indices(self.nrows)))
        except TypeError:
            raise TypeError(
                'expected `index` is integer or slice not'
                ' {}'.format(type(index)))
        return self.__class__(submatrix, self._ncolumns)

    def __repr__(self):
        """Return string representation of matrix to use in terminal."""
        rep = '{name}(shapes={shapes}, [{{matrix}}])'.format(
            name=self.__class__.__name__,
            shapes=self.shapes
            )
        if not self._matrix:
            return rep.format(matrix='')
        matrix = ''
        if self.nrows <= 3:
            for i, vec in enumerate(self._matrix):
                str_vec = str(vec)
                if len(str_vec) > 8:
                    str_vec = '{first4}...{last4}'.format(
                        first4=str_vec[:4], last4=str_vec[-4:])
                matrix += '{}: {}, '.format(str(i), str_vec)
        else:
            for i, vec in [(0, self._matrix[0]),
                           (1, self._matrix[1]),
                           (self.nrows - 1, self._matrix[-1])]:
                str_vec = str(vec)
                if len(str_vec) > 8:
                    str_vec = '{first4}...{last4}'.format(
                        first4=str_vec[:4], last4=str_vec[-4:])
                if i == 1:
                    matrix += '{}: {}, ..., '.format(str(i), str_vec)
                else:
                    matrix += '{}: {}, '.format(str(i), str_vec)
        return rep.format(matrix=matrix[:-2])

    def __str__(self):
        """Return string representation of Matrix to print it."""
        return self.to_str()

    def __setitem__(self, index, row):
        """Set the row with index `index` by new `row`."""
        if not isinstance(index, int):
            raise TypeError(
                'excepted `index` is integer, but got {} '
                ''.format(type(index)))
        if abs(index) >= self.nrows:
            raise IndexError(
                'assignment index out of range,'
                ' expected |index| < {}'.format(self.nrows))
        index = index % self.nrows
        self._matrix = (self._matrix[:index] +
                        (self.__make_row_from_value(row), ) +
                        self._matrix[index + 1:])

    def __eq__(self, other):
        """Return True if self == other."""
        try:
            if self.nrows != other.nrows:
                return False
            for row1, row2 in zip(self, other):
                if row1 != row2:
                    return False
        except (AttributeError, TypeError):
            return False
        return True

    def __ne__(self, other):
        """Return False if self == other."""
        return not self == other

    def __imul__(self, other):
        """Multiply of two matrices.

        self *= other and return self.
        """
        if self.ncolumns != other.nrows:
            raise ValueError(
                'wrong shapes of matrices: the number of '
                'columns of the first matrix must be equal the '
                'number of rows of other matrix, '
                'but {} != {}'.format(self.ncolumns, other.nrows))
        result = []
        for row in self:
            sum_row = vector.Vector(0, other.ncolumns)
            for vec in (other_row for i, other_row in enumerate(other)
                        if row[i]):
                sum_row += vec
            result.append(sum_row)
        self._matrix = tuple(result)
        self._ncolumns = other.ncolumns
        return self

    def __mul__(self, other):
        """Multiply of two matrices.

        return self * other
        """
        if self.ncolumns != other.nrows:
            raise ValueError(
                'wrong shapes of matrices: the number of '
                'columns of the first matrix must be equal the '
                'number of rows of other matrix, '
                'but {} != {}'.format(self.ncolumns, other.nrows))
        result = []
        for row in self:
            sum_row = vector.Vector(0, other.ncolumns)
            for vec in (other_row for i, other_row in enumerate(other)
                        if row[i]):
                sum_row += vec
            result.append(sum_row.value)
        return self.__class__(result, other.ncolumns)

    def __iadd__(self, other):
        """Sum of two matrices.

        self += other and return self.
        """
        result = []
        for row1, row2 in zip(self, other):
            result.append(row1 + row2)
        self._matrix = tuple(result)
        self._ncolumns = max(self.ncolumns, other.ncolumns)
        return self

    def __add__(self, other):
        """Sum of two matrices.

        return self + other
        """
        return self.__class__(
            tuple((row1 + row2).value for row1, row2 in zip(self, other)),
            max(self.ncolumns, other.ncolumns))

    def __ixor__(self, other):
        """Evaluate XOR of two matrices.

        self ^= other and return self.
        """
        self += other
        return self

    def __xor__(self, other):
        """Evaluate XOR of two matrices.

        return self ^ other
        """
        return self + other

    def __ior__(self, other):
        """Evaluate OR of two matrices.

        self |= other and return self.
        """
        result = []
        for row1, row2 in zip(self, other):
            result.append(row1 | row2)
        self._matrix = tuple(result)
        self._ncolumns = max(self.ncolumns, other.ncolumns)
        return self

    def __or__(self, other):
        """Evaluate OR of two matrices.

        return self ^ other
        """
        return self.__class__(
            tuple((row1 | row2).value for row1, row2 in zip(self, other)),
            max(self.ncolumns, other.ncolumns))

    def __iand__(self, other):
        """Evaluate AND of two matrices.

        self &= other and return self.
        """
        result = []
        for row1, row2 in zip(self, other):
            result.append(row1 & row2)
        self._matrix = tuple(result)
        self._ncolumns = max(self.ncolumns, other.ncolumns)
        return self

    def __and__(self, other):
        """Evaluate AND of two matrices.

        return self & other
        """
        return self.__class__(
            tuple((row1 & row2).value for row1, row2 in zip(self, other)),
            max(self.ncolumns, other.ncolumns))

    def __make_row_from_value(self, value):
        """Make row from value of various type."""
        try:
            value = value.value
            new_row = vector.Vector(value, self._ncolumns)
        except AttributeError:
            if isinstance(value, int):
                new_row = vector.Vector(value, self._ncolumns)
            elif isinstance(value, str):
                new_row = vector.from_string(value)
                new_row.set_length(self._ncolumns)
            else:
                new_row = vector.from_iterable(value)
                new_row.set_length(self._ncolumns)
        return new_row


def from_vectors(vectors):
    """Return matrix from vectors list."""
    return Matrix(
        (vec.value for vec in vectors),
        max((len(vec) for vec in vectors)))


def from_string(value, zerofillers=None, onefillers=None, row_sep=';'):
    """Make Matrix object from string `value`."""
    if not value:
        return Matrix()
    try:
        row_str_list = [lex for lex in value.split(row_sep) if lex != '']
    except AttributeError:
        raise TypeError(
            'expected `value` is string, but got '
            '{}'.format(type(value)))
    return Matrix(
        (vector.from_string(
            row,
            onefillers=onefillers,
            zerofillers=zerofillers).value for row in row_str_list),
        max(len(s) for s in row_str_list))


def from_iterable(value, zerofillers=None, onefillers=None):
    """Make Matrix object from list of iterable `value`."""
    if not value:
        return Matrix()
    matrix_rows = tuple(
        vector.from_iterable(
            row,
            onefillers=onefillers,
            zerofillers=zerofillers) for row in value)
    return Matrix(
        (row.value for row in matrix_rows),
        max(len(row) for row in matrix_rows))


def zero(nrows, ncolumns=None):
    """Return (nrows x ncolumns)-matrix of zeroes."""
    if not ncolumns:
        ncolumns = nrows
    return Matrix([0] * nrows, ncolumns)


def identity(nrows, ncolumns=None):
    """Return (nrows x ncolumns) identity matrix."""
    if not ncolumns:
        ncolumns = nrows
    return Matrix(
        (1 << (ncolumns - i - 1) for i in range(min(nrows, ncolumns))),
        ncolumns)


def random(nrows, ncolumns=None, max_rank=False):
    """Return random matrix."""
    if not ncolumns:
        ncolumns = nrows
    if not max_rank:
        return Matrix(
            (randint(1, (1 << ncolumns) - 1) for _ in range(nrows)),
            ncolumns)
    if nrows == ncolumns:
        return nonsingular(nrows)
    nonsing = nonsingular(min(nrows, ncolumns))
    perm_matrix = permutation(sample(range(max(nrows, ncolumns)),
                                     max(nrows, ncolumns)))
    if nrows < ncolumns:
        return concatenate(nonsing, random(ncolumns - nrows)) * perm_matrix
    return perm_matrix * concatenate(nonsing,
                                     random(nrows - ncolumns),
                                     by_rows=True)


def nonsingular(size):
    """Return non-singular binary square matrix.

    Function uses algorithm of Dana Randall
    https://www.researchgate.net/publication/2729950_Efficient_Generation_of_Random_Nonsingular_Matrices
    """
    size = max(0, size)
    matr_a_rows = [vector.Vector(0, size) for i in range(size)]
    matr_t_rows = {}
    restricted = []
    for i in range(size):
        vec_v = vector.Vector(randint(1, (1 << (size - i)) - 1), size - i)
        for j, bit in enumerate(vec_v):
            if bit:
                val_r = [k for k in range(size) if k not in restricted][j]
                break
        matr_a_rows[i][val_r] = 1
    # There is mistake in the paper of Dana Randall: this code was missing
    # in her paper
        for k in range(i+1, size):
            matr_a_rows[k][val_r] = randint(0, 1)
    # --------------------------------------------------------------------
        matr_t_rows[val_r] = vector.Vector(0, size)
        for j, k in enumerate(j for j in range(size) if j not in restricted):
            matr_t_rows[val_r][k] = vec_v[j]
        restricted.append(val_r)
    return Matrix(
        (row.value for row in matr_a_rows), size) * Matrix(
            (matr_t_rows[i].value for i in sorted(matr_t_rows)), size)


def concatenate(first, second, by_rows=False):
    """Concatenate two matrices."""
    if by_rows:
        return Matrix(
            (row.value for row in tuple(first) + tuple(second)),
            max(first.ncolumns, second.ncolumns))
    return Matrix(
        (vector.concatenate(row_first, row_second).value
         for row_first, row_second in zip(first, second)),
        first.ncolumns + second.ncolumns)


def permutation(perm, by_rows=False):
    """Return square matrix represented the permutation `perm`."""
    ncolumns = len(perm)
    if by_rows:
        row_values = ((1 << (ncolumns - 1 - i)) for i in perm)
    else:
        row_values = [0]*ncolumns
        for i, j in enumerate(perm):
            row_values[j] = (1 << (ncolumns - 1 - i))
    return Matrix(row_values, ncolumns)
