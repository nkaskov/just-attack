"""Unit tests for matrix module."""

import unittest
from blincodes import matrix
from blincodes.vector import Vector


class InitMatrixTestCase(unittest.TestCase):
    """Test to init of Matrix object."""

    def test_init_default(self):
        """Init by default values."""
        matr = matrix.Matrix()
        self.assertEqual(matr.shapes, (0, 0))
        self.assertEqual(list(matr), [])

    def test_init_by_integers(self):
        """Init by list of integers."""
        matr = matrix.Matrix((0, 0b0011, 0b1011))
        self.assertEqual(matr.shapes, (0, 0))
        self.assertEqual(list(matr), [])

        matr = matrix.Matrix((0, 0b0011, 0b1011), ncolumns=2)
        self.assertEqual(matr.shapes, (3, 2))
        self.assertEqual(list(matr),
                         [Vector(0, 2), Vector(0b0011, 2), Vector(0b1011, 2)])

        matr = matrix.Matrix((0, 0b0011, 0b1011), ncolumns=4)
        self.assertEqual(matr.shapes, (3, 4))
        self.assertEqual(list(matr),
                         [Vector(0, 4), Vector(0b0011, 4), Vector(0b1011, 4)])

        matr = matrix.Matrix((0, 0b0011, 0b1011), ncolumns=10)
        self.assertEqual(matr.shapes, (3, 10))
        self.assertEqual(list(matr),
                         [Vector(0, 10),
                          Vector(0b0011, 10),
                          Vector(0b1011, 10)])

    def test_init_by_vectors(self):
        """Init by list of vectors."""
        matr = matrix.from_vectors([Vector(0b011, 3),
                                    Vector(0b1110, 4),
                                    Vector(0b01, 2)])
        self.assertEqual(matr.shapes, (3, 4))
        self.assertEqual(list(matr), [Vector(0b011, 4),
                                      Vector(0b1110, 4),
                                      Vector(0b01, 4)])

    def test_init_by_string(self):
        """Init Matrix object by string."""
        matr = matrix.from_string(
            '10000101;'
            '01001;'
            '00011100101;'
            '0101001'
            )
        self.assertEqual(matr.shapes, (4, 11))
        self.assertEqual(list(matr), [Vector(0b00010000101, 11),
                                      Vector(0b00000001001, 11),
                                      Vector(0b00011100101, 11),
                                      Vector(0b00000101001, 11)])

        matr = matrix.from_string('')
        self.assertEqual(matr.shapes, (0, 0))
        self.assertEqual(list(matr), [])

        matr = matrix.from_string(
            '100**101\\'
            '0100|\\'
            '00-..10-101\\'
            '01$1-01',
            zerofillers='*$-',
            onefillers='|.',
            row_sep='\\'
            )
        self.assertEqual(matr.shapes, (4, 11))
        self.assertEqual(list(matr), [Vector(0b00010000101, 11),
                                      Vector(0b00000001001, 11),
                                      Vector(0b00011100101, 11),
                                      Vector(0b00000101001, 11)])

    def test_init_by_iterable(self):
        """Init Matrix object by iterable."""
        matr = matrix.from_iterable(
            [
                ('*', 1, '&', 1, 1, '-', '0', '1', 1, 0, '|*1-'),
                ('*', 1, '&', '-', '0', '1', 1, 0, '|*1-'),
                ('*', 1, '&', 1, 1, '-', '0', '1', 1, 0, '|*1-00111101'),
            ],
            onefillers='&|',
            zerofillers='*-'
            )
        self.assertEqual(matr.shapes, (3, 22))
        self.assertEqual(list(matr), [Vector(0b0000000001111001101010, 22),
                                      Vector(0b0000000000011001101010, 22),
                                      Vector(0b0111100110101000111101, 22)])

        matr = matrix.from_iterable([])
        self.assertEqual(matr.shapes, (0, 0))
        self.assertEqual(list(matr), [])

        matr = matrix.from_iterable([[], []])
        self.assertEqual(matr.shapes, (0, 0))
        self.assertEqual(list(matr), [])


class AriphmeticsAndComparingMatrixTestCase(unittest.TestCase):
    """Testing arithmetics and comparing functions."""

    def test_bool(self):
        """Test comparing with True and False."""
        self.assertFalse(matrix.Matrix())
        self.assertTrue(matrix.Matrix([0b1], ncolumns=1))

    def test_iterator(self):
        """Test iteration over matrix rows."""
        self.assertEqual(list(matrix.Matrix()), [])
        self.assertTrue(list(matrix.Matrix([0b1], ncolumns=1)),
                        [Vector(0b1, 1)])
        self.assertTrue(list(matrix.Matrix(
            [0b0011, 0b1010, 0b0111], ncolumns=4)),
                        [Vector(0b0011, 4),
                         Vector(0b1010, 4),
                         Vector(0b0111, 4)])

    def test_getitem(self):
        """Test getting the item."""
        matr = matrix.Matrix(
            [
                0b11110000101,
                0b01100001001,
                0b00011100101,
                0b10100101001,
            ],
            ncolumns=11)
        self.assertIsInstance(matr[2], Vector)
        self.assertEqual(matr[2], Vector(0b00011100101, 11))
        self.assertEqual(matr[-2], Vector(0b00011100101, 11))
        self.assertIsInstance(matr[0:4:2], matrix.Matrix)
        self.assertEqual(list(matr[0:4:2]),
                         [Vector(0b11110000101, 11),
                          Vector(0b00011100101, 11)])
        self.assertIsInstance(matr[::-1], matrix.Matrix)
        self.assertEqual(list(matr[::-1]),
                         [Vector(0b10100101001, 11),
                          Vector(0b00011100101, 11),
                          Vector(0b01100001001, 11),
                          Vector(0b11110000101, 11)])

    def test_setitem(self):
        """Test setting the item."""
        matr_values = [
            0b11110000101,
            0b01100001001,
            0b00011100101,
            0b10100101001,
            ]
        matr = matrix.Matrix(matr_values, ncolumns=11)
        matr[0] = 0
        self.assertEqual([x.value for x in matr], [0] + matr_values[1:])
        matr = matrix.Matrix(matr_values, ncolumns=11)
        matr[0] = Vector(0b11, 3)
        self.assertEqual([x.value for x in matr], [0b011] + matr_values[1:])
        matr = matrix.Matrix(matr_values, ncolumns=11)
        matr[0] = '1011'
        self.assertEqual([x.value for x in matr], [0b1011] + matr_values[1:])
        matr = matrix.Matrix(matr_values, ncolumns=11)
        matr[0] = (1, 0, '11', '001', True)
        self.assertEqual([x.value for x in matr],
                         [0b10110011] + matr_values[1:])
        matr = matrix.Matrix(matr_values, ncolumns=11)
        matr[2] = 0
        self.assertEqual([x.value for x in matr],
                         matr_values[:2] + [0] + matr_values[3:])
        matr[-2] = 0
        self.assertEqual([x.value for x in matr],
                         matr_values[:2] + [0] + matr_values[3:])

    def test_equality(self):
        """Test matrix's equality."""
        matr_values = [
            0b11110000101,
            0b01100001001,
            0b00011100101,
            0b10100101001,
            ]
        matr = matrix.Matrix(matr_values, ncolumns=11)
        self.assertEqual(matr, matrix.Matrix(matr_values, ncolumns=11))
        self.assertNotEqual(matr, matrix.Matrix())
        self.assertEqual(matrix.Matrix(), matrix.Matrix())
        matr[0] = 0
        self.assertNotEqual(matrix.Matrix(matr_values, ncolumns=11), matr)
        self.assertNotEqual(matrix.Matrix(matr_values, ncolumns=11),
                            matrix.Matrix([0b11, 0b01], 2))

    def test_multiplication(self):
        """Test multiply of matrices."""
        matr_values = [
            0b11110000101,
            0b01100001001,
            0b00011100101,
            0b10100101001,
        ]
        matr_a_values = [
            0b1001,
            0b1100,
            0b1101,
            0b1010,
            0b0010,
            0b0011,
            0b0000,
            0b0101,
            0b1010,
            0b0000,
            0b1111,
        ]
        matr_result = [
            0b0111,
            0b1011,
            0b1110,
            0b1101,
        ]
        self.assertEqual(
            matrix.Matrix(matr_values, 11) * matrix.Matrix(matr_a_values, 4),
            matrix.Matrix(matr_result, 4))
        self.assertEqual(matrix.Matrix() * matrix.Matrix(), matrix.Matrix())
        with self.assertRaises(ValueError):
            matrix.Matrix(matr_values, 11) * matrix.Matrix()
        matr_a = matrix.Matrix(matr_values, 11)
        matr_a *= matrix.Matrix(matr_a_values, 4)
        self.assertEqual(matr_a, matrix.Matrix(matr_result, 4))

    def test_addition(self):
        """Test add of matrices."""
        matr_values = [
            0b11110000101,
            0b01100001001,
            0b00011100101,
            0b10100101001,
        ]
        matr_a_values = [
            0b1001,
            0b1100,
            0b1101,
            0b1010,
            0b0010,
            0b0011,
            0b0000,
            0b0101,
            0b1010,
            0b0000,
            0b1111,
        ]
        matr_result = [
            0b11110001100,
            0b01100000101,
            0b00011101000,
            0b10100100011,
        ]
        self.assertEqual(
            matrix.Matrix(matr_values, 11) + matrix.Matrix(matr_values, 11),
            matrix.Matrix([0] * 4, 11))
        self.assertEqual(matrix.Matrix() + matrix.Matrix(), matrix.Matrix())
        self.assertEqual(
            matrix.Matrix(matr_values, 11) + matrix.Matrix(),
            matrix.Matrix())
        self.assertEqual(
            matrix.Matrix(matr_values, 11) + matrix.Matrix(matr_a_values, 4),
            matrix.Matrix(matr_result, 11))
        matr_a = matrix.Matrix(matr_values, 11)
        matr_a += matrix.Matrix(matr_a_values, 11)
        self.assertEqual(matr_a, matrix.Matrix(matr_result, 11))

    def test_xor(self):
        """Test xor of matrices."""
        self.test_addition()

    def test_or(self):
        """Test OR operation under matrices."""
        matr_values = [
            0b11110000101,
            0b01100001001,
            0b00011100101,
            0b10100101001,
        ]
        matr_a_values = [
            0b1001,
            0b1100,
            0b1101,
            0b1010,
            0b0010,
            0b0011,
            0b0000,
            0b0101,
            0b1010,
            0b0000,
            0b1111,
        ]
        matr_result = [
            0b11110001101,
            0b01100001101,
            0b00011101101,
            0b10100101011,
        ]
        self.assertEqual(
            matrix.Matrix(matr_values, 11) | matrix.Matrix(matr_values, 11),
            matrix.Matrix(matr_values, 11))
        self.assertEqual(matrix.Matrix() | matrix.Matrix(), matrix.Matrix())
        self.assertEqual(
            matrix.Matrix(matr_values, 11) | matrix.Matrix(),
            matrix.Matrix())
        self.assertEqual(
            matrix.Matrix(matr_values, 11) | matrix.Matrix(matr_a_values, 4),
            matrix.Matrix(matr_result, 11))
        matr_a = matrix.Matrix(matr_values, 11)
        matr_a |= matrix.Matrix(matr_a_values, 11)
        self.assertEqual(matr_a, matrix.Matrix(matr_result, 11))

    def test_and(self):
        """Test AND operation under matrices."""
        matr_values = [
            0b11110000101,
            0b01100001001,
            0b00011100101,
            0b10100101001,
        ]
        matr_a_values = [
            0b1001,
            0b1100,
            0b1101,
            0b1010,
            0b0010,
            0b0011,
            0b0000,
            0b0101,
            0b1010,
            0b0000,
            0b1111,
        ]
        matr_result = [
            0b0001,
            0b1000,
            0b0101,
            0b1000,
        ]
        self.assertEqual(
            matrix.Matrix(matr_values, 11) & matrix.Matrix(matr_values, 11),
            matrix.Matrix(matr_values, 11))
        self.assertEqual(matrix.Matrix() & matrix.Matrix(), matrix.Matrix())
        self.assertEqual(
            matrix.Matrix(matr_values, 11) & matrix.Matrix(),
            matrix.Matrix())
        self.assertEqual(
            matrix.Matrix(matr_values, 11) & matrix.Matrix(matr_a_values, 4),
            matrix.Matrix(matr_result, 11))
        matr_a = matrix.Matrix(matr_values, 11)
        matr_a &= matrix.Matrix(matr_a_values, 11)
        self.assertEqual(matr_a, matrix.Matrix(matr_result, 11))


class StringRepresentationMatrixTestCase(unittest.TestCase):
    """Testing representation as string."""

    def test_to_str(self):
        """Test representation as customisable string."""
        matr_values = [
            0b111111111111,
            0b011111111111,
            0b001111111111,
            0b000111111111,
            0b000011111111,
            0b000001111111,
            0b000000111111,
            0b000000011111,
            0b000000001111,
            0b000000000111,
            0b000000000011,
            0b000000000001
        ]
        matr_str = (
            '111111111111\n'
            '011111111111\n'
            '001111111111\n'
            '000111111111\n'
            '000011111111\n'
            '000001111111\n'
            '000000111111\n'
            '000000011111\n'
            '000000001111\n'
            '000000000111\n'
            '000000000011\n'
            '000000000001')
        matr_str_numbered = (
            ' 0: 111111111111\n'
            ' 1: 011111111111\n'
            ' 2: 001111111111\n'
            ' 3: 000111111111\n'
            ' 4: 000011111111\n'
            ' 5: 000001111111\n'
            ' 6: 000000111111\n'
            ' 7: 000000011111\n'
            ' 8: 000000001111\n'
            ' 9: 000000000111\n'
            '10: 000000000011\n'
            '11: 000000000001')
        self.assertEqual(
            matrix.Matrix(matr_values, 12).to_str(), matr_str)
        self.assertEqual(
            matrix.Matrix(matr_values, 12).to_str(zerofillers='*'),
            matr_str.replace('0', '*'))
        self.assertEqual(
            matrix.Matrix(matr_values, 12).to_str(onefillers='*'),
            matr_str.replace('1', '*'))
        self.assertEqual(
            matrix.Matrix().to_str(onefillers='*'),
            '')
        self.assertEqual(
            matrix.Matrix(matr_values, 12).to_str(numbered=True),
            matr_str_numbered)

    def test_to_str_default(self):
        """Test representation as string to print it."""
        matr_values = [
            0b111111111111,
            0b011111111111,
            0b001111111111,
            0b000111111111,
            0b000011111111,
            0b000001111111,
            0b000000111111,
            0b000000011111,
            0b000000001111,
            0b000000000111,
            0b000000000011,
            0b000000000001
        ]
        matr_str = (
            '111111111111\n'
            '011111111111\n'
            '001111111111\n'
            '000111111111\n'
            '000011111111\n'
            '000001111111\n'
            '000000111111\n'
            '000000011111\n'
            '000000001111\n'
            '000000000111\n'
            '000000000011\n'
            '000000000001')
        self.assertEqual(
            str(matrix.Matrix(matr_values, 12)), matr_str)
        self.assertEqual(
            str(matrix.Matrix()),
            '')

    def test_to_str_repr(self):
        """Test representation as string."""
        matr_values = [
            0b111111111111,
            0b011111111111,
            0b001111111111,
            0b000111111111,
            0b000011111111,
            0b000001111111,
            0b000000111111,
            0b000000011111,
            0b000000001111,
            0b000000000111,
            0b000000000011,
            0b000000000001
        ]
        matr_str = (
            'Matrix(shapes=(12, 12), ['
            '0: 1111...1111, '
            '1: 0111...1111, '
            '..., '
            '11: 0000...0001])')
        matr_str8 = (
            'Matrix(shapes=(12, 8), ['
            '0: 11111111, '
            '1: 11111111, '
            '..., '
            '11: 00000001])')
        matr_str48 = (
            'Matrix(shapes=(3, 8), ['
            '0: 11111111, '
            '1: 11111111, '
            '2: 11111111])')
        self.assertEqual(
            repr(matrix.Matrix(matr_values, 12)), matr_str)
        self.assertEqual(
            repr(matrix.Matrix(matr_values, 8)), matr_str8)
        self.assertEqual(
            repr(matrix.Matrix(matr_values, 8)[:3]), matr_str48)
        self.assertEqual(
            repr(matrix.Matrix()),
            'Matrix(shapes=(0, 0), [])')

    def test_to_latex_str(self):
        """Test representation Matrix object as LaTeX string."""
        matr_values = [
            0b111111111111,
            0b011111111111,
            0b001111111111,
            0b000111111111,
            0b000011111111,
            0b000001111111,
            0b000000111111,
            0b000000011111,
            0b000000001111,
            0b000000000111,
            0b000000000011,
        ]
        matr_str = (
            '1&1&1&1&1&1&1&1&1&1&1&1\\\\\n'
            '0&1&1&1&1&1&1&1&1&1&1&1\\\\\n'
            '0&0&1&1&1&1&1&1&1&1&1&1\\\\\n'
            '0&0&0&1&1&1&1&1&1&1&1&1\\\\\n'
            '0&0&0&0&1&1&1&1&1&1&1&1\\\\\n'
            '0&0&0&0&0&1&1&1&1&1&1&1\\\\\n'
            '0&0&0&0&0&0&1&1&1&1&1&1\\\\\n'
            '0&0&0&0&0&0&0&1&1&1&1&1\\\\\n'
            '0&0&0&0&0&0&0&0&1&1&1&1\\\\\n'
            '0&0&0&0&0&0&0&0&0&1&1&1\\\\\n'
            '0&0&0&0&0&0&0&0&0&0&1&1')
        matr = matrix.Matrix(matr_values, 12)
        self.assertEqual(matr.to_latex_str(), matr_str)
        self.assertEqual(matrix.Matrix().to_latex_str(), '')

    def test_shapes(self):
        """Test get shapes."""
        matr_values = [
            0b111111111111,
            0b011111111111,
            0b001111111111,
            0b000111111111,
            0b000011111111,
            0b000001111111,
            0b000000111111,
            0b000000011111,
            0b000000001111,
            0b000000000111,
            0b000000000011,
        ]
        matr = matrix.Matrix(matr_values, 12)
        self.assertEqual(matr.shapes, (11, 12))
        self.assertEqual(matr.nrows, 11)
        self.assertEqual(matr.ncolumns, 12)
        matr_empty = matrix.Matrix()
        self.assertEqual(matr_empty.shapes, (0, 0))
        self.assertEqual(matr_empty.ncolumns, 0)
        self.assertEqual(matr_empty.nrows, 0)

    def test_make_copy(self):
        """Test make copy of Matrix object."""
        matr_values = [
            0b111111111111,
            0b011111111111,
            0b001111111111,
            0b000111111111,
            0b000011111111,
            0b000001111111,
            0b000000111111,
            0b000000011111,
            0b000000001111,
            0b000000000111,
            0b000000000011,
        ]
        matr = matrix.Matrix(matr_values, 12)
        self.assertEqual(matr.copy(), matr)
        self.assertEqual(matrix.Matrix().copy(), matrix.Matrix())

    def test_submatrix(self):
        """Test choice submatrix."""
        matr_values = [
            0b111111111111,
            0b011111111111,
            0b001111111111,
            0b000111111111,
            0b000011111111,
            0b000001111111,
            0b000000111111,
            0b000000011111,
            0b000000001111,
            0b000000000111,
            0b000000000011,
        ]
        submatr_values = [
            0b1111111,
            0b1111111,
            0b0111111,
            0b0111111,
            0b0011111,
            0b0001111,
            0b0001111,
            0b0001111,
            0b0000111,
            0b0000011,
            0b0000001,
        ]
        submatr_values2 = [
            0b1111111111,
            0b1111111011,
            0b0111111011,
            0b0111111011,
            0b0011111011,
            0b0001111011,
            0b0001111010,
            0b0001111010,
            0b0000111010,
            0b0000011000,
            0b0000001000,
        ]
        matr = matrix.Matrix(matr_values, 12)
        self.assertEqual(matr.submatrix([1, 3, 4, 7, 8, 9, -1]),
                         matrix.Matrix(submatr_values, 7))
        self.assertEqual(matr.submatrix(),
                         matr)
        self.assertEqual(matr.submatrix([1, 3, 4, 7, 8, 9, -1, 12, 20, 17]),
                         matrix.Matrix(submatr_values2, 10))
        self.assertEqual(matrix.Matrix().submatrix([0, 7, 8]),
                         matrix.Matrix())

    def test_transpose(self):
        """Test matrix transposition."""
        matr_values = [
            0b111111111111,
            0b011111111111,
            0b001111111111,
            0b000111111111,
            0b000011111111,
            0b000001111111,
            0b000000111111,
            0b000000011111,
            0b000000001111,
            0b000000000111,
            0b000000000011,
        ]
        transpose_values = [
            0b10000000000,
            0b11000000000,
            0b11100000000,
            0b11110000000,
            0b11111000000,
            0b11111100000,
            0b11111110000,
            0b11111111000,
            0b11111111100,
            0b11111111110,
            0b11111111111,
            0b11111111111
        ]
        matr = matrix.Matrix(matr_values, 12)
        self.assertEqual(matr.transpose(),
                         matrix.Matrix(transpose_values, 11))
        self.assertEqual(matr.transpose().transpose(), matr)
        self.assertEqual(matrix.Matrix().transpose(),
                         matrix.Matrix())
        self.assertEqual(matr.T, matr.transpose())

    def test_concatenate(self):
        """Test matrix concatenation."""
        matr_values1 = [
            0b111111111111,
            0b011111111111,
            0b001111111111,
            0b000111111111,
            0b000011111111,
            0b000001111111,
            0b000000111111,
            0b000000011111,
            0b000000001111,
            0b000000000111,
            0b000000000011,
        ]
        matr_values2 = [
            0b10000000000,
            0b11000000000,
            0b11100000000,
            0b11110000000,
            0b11111000000,
            0b11111100000,
            0b11111110000,
            0b11111111000,
            0b11111111100,
            0b11111111110,
            0b11111111111,
            0b11111111111
        ]
        matr_concat_columns = [
            0b11111111111110000000000,
            0b01111111111111000000000,
            0b00111111111111100000000,
            0b00011111111111110000000,
            0b00001111111111111000000,
            0b00000111111111111100000,
            0b00000011111111111110000,
            0b00000001111111111111000,
            0b00000000111111111111100,
            0b00000000011111111111110,
            0b00000000001111111111111,
        ]
        matr_concat_rows = [
            0b111111111111,
            0b011111111111,
            0b001111111111,
            0b000111111111,
            0b000011111111,
            0b000001111111,
            0b000000111111,
            0b000000011111,
            0b000000001111,
            0b000000000111,
            0b000000000011,
            0b010000000000,
            0b011000000000,
            0b011100000000,
            0b011110000000,
            0b011111000000,
            0b011111100000,
            0b011111110000,
            0b011111111000,
            0b011111111100,
            0b011111111110,
            0b011111111111,
            0b011111111111
        ]
        matr1 = matrix.Matrix(matr_values1, 12)
        matr2 = matrix.Matrix(matr_values2, 11)
        self.assertEqual(matrix.concatenate(matr1, matr2),
                         matrix.Matrix(matr_concat_columns, 23))
        self.assertEqual(matr1.concatenate(matr2),
                         matrix.Matrix(matr_concat_columns, 23))
        matr1 = matrix.Matrix(matr_values1, 12)
        self.assertEqual(matrix.concatenate(matr1, matr2, by_rows=True),
                         matrix.Matrix(matr_concat_rows, 12))
        self.assertEqual(matr1.concatenate(matr2, by_rows=True),
                         matrix.Matrix(matr_concat_rows, 12))
        matr1 = matrix.Matrix(matr_values1, 12)
        self.assertEqual(matrix.concatenate(matrix.Matrix(), matr1),
                         matrix.Matrix())
        self.assertEqual(matrix.Matrix().concatenate(matr1),
                         matrix.Matrix())
        self.assertEqual(matrix.concatenate(matr1, matrix.Matrix()),
                         matrix.Matrix())
        self.assertEqual(matr1.concatenate(matrix.Matrix()),
                         matr1)
        matr1 = matrix.Matrix(matr_values1, 12)
        self.assertEqual(matrix.concatenate(matrix.Matrix(),
                                            matr1, by_rows=True),
                         matr1)
        self.assertEqual(matrix.Matrix().concatenate(matr1, by_rows=True),
                         matr1)
        self.assertEqual(matrix.concatenate(matr1,
                                            matrix.Matrix(), by_rows=True),
                         matr1)
        self.assertEqual(matr1.concatenate(matrix.Matrix(), by_rows=True),
                         matr1)

    def test_is_zero(self):
        """Test matrix comparing with zero."""
        matr_values = [
            0b111111111111,
            0b011111111111,
            0b001111111111,
            0b000111111111,
            0b000011111111,
            0b000001111111,
            0b000000111111,
            0b000000011111,
            0b000000001111,
            0b000000000111,
            0b000000000011,
        ]
        self.assertFalse(matrix.Matrix(matr_values, 12).is_zero())
        self.assertTrue(matrix.Matrix([0] * 15, 12).is_zero())
        self.assertTrue(matrix.Matrix().is_zero())

    def test_is_identity(self):
        """Test matrix comparing with identity matrix."""
        matr_values = [
            0b111111111111,
            0b011111111111,
            0b001111111111,
            0b000111111111,
            0b000011111111,
            0b000001111111,
            0b000000111111,
            0b000000011111,
            0b000000001111,
            0b000000000111,
            0b000000000011,
        ]
        self.assertFalse(matrix.Matrix(matr_values, 12).is_identity())
        self.assertTrue(
            matrix.Matrix([1 << (11 - i)
                           for i in range(12)], 12).is_identity())
        self.assertFalse(matrix.Matrix().is_identity())


class MatrixLinearTransformationsTestCase(unittest.TestCase):
    """Testing linear transformation of matrix and solving linear equations."""

    def setUp(self):
        """Set the test value."""
        self.matr_upper = [
            0b1111,
            0b0111,
            0b0011,
            0b0001,
        ]
        self.matr_max_rank = [
            0b0111,
            0b1000,
            0b1100,
            0b1110,
        ]
        self.matr_non_max_rank1 = [
            0b01110,
            0b00101,
            0b11001,
            0b11100,
        ]
        self.matr_non_max_rank2 = [
            0b01110,
            0b00101,
            0b11001,
            0b11100,
            0b10010,
            0b11111,
            0b01010,
        ]

    def test_echelon_form(self):
        """Test  evaluating of matrix echelon form."""
        matr_max_rank_echelon = [
            0b1000,
            0b0111,
            0b0011,
            0b0001,
        ]
        matr_non_max_rank1_echelon = [
            0b10010,
            0b01110,
            0b00101,
            0b00000,
        ]
        matr_non_max_rank2_echelon = [
            0b10010,
            0b01110,
            0b00101,
            0b00011,
            0b00001,
            0b00000,
            0b00000,
        ]
        self.assertEqual(matrix.Matrix(self.matr_upper, 4).echelon_form,
                         matrix.Matrix(self.matr_upper, 4))
        self.assertEqual(matrix.Matrix(self.matr_max_rank, 4).echelon_form,
                         matrix.Matrix(matr_max_rank_echelon, 4))
        self.assertEqual(matrix.Matrix(self.matr_non_max_rank1,
                                       5).echelon_form,
                         matrix.Matrix(matr_non_max_rank1_echelon, 5))
        self.assertEqual(matrix.Matrix(self.matr_non_max_rank2,
                                       5).echelon_form,
                         matrix.Matrix(matr_non_max_rank2_echelon, 5))
        self.assertEqual(matrix.Matrix().echelon_form,
                         matrix.Matrix())

    def test_rank(self):
        """Test evaluating of matrix rank."""
        self.assertEqual(matrix.Matrix(self.matr_upper, 4).rank, 4)
        self.assertEqual(matrix.Matrix(self.matr_max_rank, 4).rank, 4)
        self.assertEqual(matrix.Matrix(self.matr_non_max_rank1,
                                       5).rank, 3)
        self.assertEqual(matrix.Matrix(self.matr_non_max_rank2,
                                       5).rank, 5)
        self.assertEqual(matrix.Matrix().rank, 0)

    def test_is_max_rank(self):
        """Test check if matrix has maximal rank."""
        self.assertTrue(matrix.Matrix(self.matr_upper, 4).is_max_rank())
        self.assertTrue(matrix.Matrix(self.matr_max_rank, 4).is_max_rank())
        self.assertFalse(matrix.Matrix(self.matr_non_max_rank1,
                                       5).is_max_rank())
        self.assertTrue(matrix.Matrix(self.matr_non_max_rank2,
                                      5).is_max_rank())
        self.assertTrue(matrix.Matrix().is_max_rank)

    def test_diagonal_form(self):
        """Test evaluating of matrix diagonal form."""
        matr_non_max_rank1_diagonal = [
            0b10010,
            0b01011,
            0b00101,
            0b00000,
        ]
        matr_non_max_rank2_diagonal = [
            0b10000,
            0b01000,
            0b00100,
            0b00010,
            0b00001,
            0b00000,
            0b00000,
        ]
        self.assertTrue(
            matrix.Matrix(self.matr_upper, 4).diagonal_form.is_identity())
        self.assertTrue(
            matrix.Matrix(self.matr_max_rank, 4).diagonal_form.is_identity())
        self.assertEqual(matrix.Matrix(self.matr_non_max_rank1,
                                       5).diagonal_form,
                         matrix.Matrix(matr_non_max_rank1_diagonal, 5))
        self.assertEqual(matrix.Matrix(self.matr_non_max_rank2,
                                       5).diagonal_form,
                         matrix.Matrix(matr_non_max_rank2_diagonal, 5))
        self.assertEqual(matrix.Matrix().diagonal_form,
                         matrix.Matrix())

    def test_inverse(self):
        """Test evaluating of inverse matrix."""
        matr_non_max_rank1_diagonal = [
            0b10010,
            0b01011,
            0b00101,
            0b00000,
        ]
        matr_non_max_rank2_diagonal = [
            0b10000,
            0b01000,
            0b00100,
            0b00010,
            0b00001,
            0b00000,
            0b00000,
        ]
        self.assertTrue(
            (matrix.Matrix(self.matr_upper,
                           4).inverse * matrix.Matrix(self.matr_upper,
                                                      4)).is_identity())
        self.assertTrue(
            (matrix.Matrix(self.matr_max_rank,
                           4).inverse * matrix.Matrix(self.matr_max_rank,
                                                      4)).is_identity())
        self.assertEqual(
            matrix.Matrix(self.matr_non_max_rank1,
                          5).inverse * matrix.Matrix(self.matr_non_max_rank1,
                                                     5),
            matrix.Matrix(matr_non_max_rank1_diagonal, 5))
        self.assertEqual(
            matrix.Matrix(self.matr_non_max_rank2,
                          5).inverse * matrix.Matrix(self.matr_non_max_rank2,
                                                     5),
            matrix.Matrix(matr_non_max_rank2_diagonal, 5))
        self.assertEqual(matrix.Matrix().inverse, matrix.Matrix())
        self.assertTrue(matrix.Matrix([0] * 10, 20).inverse.is_identity())
        for _ in range(10):
            matr = matrix.nonsingular(20)
            self.assertTrue((matr * matr.inverse).is_identity())

    def test_othogonal(self):
        """Test evaluating of maximal orthogonal matrix."""
        matr_upper_ort = matrix.Matrix(self.matr_upper, 4).orthogonal
        self.assertTrue(matr_upper_ort.is_zero())
        self.assertTrue(
            (matrix.Matrix(self.matr_upper, 4) * matr_upper_ort.T).is_zero())
        matr_max_rank = matrix.Matrix(self.matr_max_rank, 4)
        matr_max_rank_ort = matr_max_rank.orthogonal
        self.assertTrue(matr_max_rank_ort.is_zero())
        self.assertTrue((matr_max_rank * matr_max_rank_ort.T).is_zero())
        matr_non_max_rank1 = matrix.Matrix(self.matr_non_max_rank1, 5)
        matr_non_max_rank1_ort = matr_non_max_rank1.orthogonal
        self.assertEqual(
            matr_non_max_rank1_ort.shapes,
            (matr_non_max_rank1.ncolumns - matr_non_max_rank1.rank,
             matr_non_max_rank1.ncolumns))
        self.assertTrue(
            (matr_non_max_rank1 * matr_non_max_rank1_ort.T).is_zero())
        matr_non_max_rank2 = matrix.Matrix(self.matr_non_max_rank2, 5)
        matr_non_max_rank2_ort = matr_non_max_rank2.orthogonal
        self.assertTrue(matr_non_max_rank2_ort.is_zero())
        self.assertTrue(
            (matr_non_max_rank2 * matr_non_max_rank2_ort.T).is_zero())
        self.assertEqual(matrix.Matrix().orthogonal, matrix.Matrix())

    def test_solving_linear_equation(self):
        """Test solving of linear equation."""
        vec = Vector(0b1010, 4)
        matr_max_rank = matrix.Matrix(self.matr_max_rank, 4)
        fundamental, vec_solve = matr_max_rank.solve(vec)
        self.assertFalse(fundamental)
        self.assertEqual(
            matr_max_rank * matrix.from_vectors([vec_solve]).transpose(),
            matrix.from_vectors([vec]).transpose())
        vec = Vector(0b1010, 4)
        matr_non_max_rank1 = matrix.Matrix(self.matr_non_max_rank1, 5)
        fundamental, vec_solve = matr_non_max_rank1.solve(vec)
        self.assertFalse(fundamental)
        self.assertFalse(vec_solve)
        vec = Vector(0b1110, 4)
        fundamental, vec_solve = matr_non_max_rank1.solve(vec)
        self.assertEqual(fundamental,
                         matrix.Matrix([0b11010, 0b01101], 5))
        self.assertEqual(
            matr_non_max_rank1 * matrix.from_vectors([vec_solve]).transpose(),
            matrix.from_vectors([vec]).transpose())

    def test_gaussian_elimination(self):
        """Test evaluating of Gaussian elimination."""
        matr_gauss_full_non_sort = [
            0b01011,
            0b00101,
            0b10010,
            0b00000,
        ]
        matr_gauss_full_sort = [
            0b10010,
            0b01011,
            0b00101,
            0b00000,
        ]
        matr_gauss_partial_non_sort = [
            0b11100,
            0b00101,
            0b10010,
            0b00000,
        ]
        matr_gauss_partial_sort = [
            0b11100,
            0b10010,
            0b00101,
            0b00000,
        ]
        matr = matrix.Matrix(self.matr_non_max_rank1, 5)
        self.assertEqual(matr.gaussian_elimination(),
                         matrix.Matrix(matr_gauss_full_sort, 5))
        self.assertEqual(matr.gaussian_elimination(sort=False),
                         matrix.Matrix(matr_gauss_full_non_sort, 5))
        self.assertEqual(matr.gaussian_elimination([1, 3, 4], sort=False),
                         matrix.Matrix(matr_gauss_partial_non_sort, 5))
        self.assertEqual(matr.gaussian_elimination([1, 3, 4]),
                         matrix.Matrix(matr_gauss_partial_sort, 5))
        self.assertEqual(matr.gaussian_elimination([1, 3, 4, -1, 7, 9, 10]),
                         matrix.Matrix(matr_gauss_partial_sort, 5))
        self.assertEqual(matr.gaussian_elimination([1, 3, 4, 4]),
                         matrix.Matrix(matr_gauss_partial_sort, 5))
        self.assertEqual(matrix.Matrix().gaussian_elimination(),
                         matrix.Matrix())


class GenerateMatrixTestCase(unittest.TestCase):
    """Testing generating of special type matrix."""

    def test_generate_zero(self):
        """Test generating zero matrix."""
        zero1 = matrix.zero(10)
        self.assertTrue(zero1.is_zero())
        self.assertEqual(zero1.shapes, (10, 10))
        zero1 = matrix.zero(20, 10)
        self.assertTrue(zero1.is_zero())
        self.assertEqual(zero1.shapes, (20, 10))
        zero1 = matrix.zero(5, 10)
        self.assertTrue(zero1.is_zero())
        self.assertEqual(zero1.shapes, (5, 10))

    def test_generate_identity(self):
        """Test generating identity matrix."""
        ident1 = matrix.identity(10)
        self.assertTrue(ident1.is_identity())
        self.assertEqual(ident1.shapes, (10, 10))
        ident1 = matrix.identity(20, 10)
        self.assertTrue(ident1.is_identity())
        self.assertEqual(ident1.shapes, (10, 10))
        ident1 = matrix.identity(5, 10)
        self.assertTrue(ident1.is_identity())
        self.assertEqual(ident1.shapes, (5, 10))

    def test_generate_random(self):
        """Test generating random matrix."""
        matr = matrix.random(10)
        self.assertEqual(matr.shapes, (10, 10))
        matr = matrix.random(10, 21)
        self.assertEqual(matr.shapes, (10, 21))
        matr = matrix.random(42, 21)
        self.assertEqual(matr.shapes, (42, 21))
        for _ in range(10):
            self.assertEqual(matrix.random(10, max_rank=True).rank, 10)
            self.assertEqual(matrix.random(10, 20, max_rank=True).rank, 10)
            self.assertEqual(matrix.random(20, 10, max_rank=True).rank, 10)

    def test_generate_nonsingular(self):
        """Test generating random non-singular matrix."""
        for _ in range(10):
            self.assertEqual(matrix.nonsingular(20).rank, 20)

    def test_generate_permutation(self):
        """Test generating permutation matrix."""
        perm_matrix = [
            0b10000000,
            0b00010000,
            0b00000010,
            0b01000000,
            0b00000100,
            0b00001000,
            0b00100000,
            0b00000001,
        ]
        self.assertEqual(matrix.permutation([]), matrix.Matrix())
        perm = matrix.permutation([0, 3, 6, 1, 5, 4, 2, 7])
        self.assertEqual(perm, matrix.Matrix(perm_matrix, 8))
        perm = matrix.permutation([0, 3, 6, 1, 5, 4, 2, 7], by_rows=True)
        self.assertEqual(perm, matrix.Matrix(perm_matrix, 8).transpose())


if __name__ == "__main__":
    unittest.main()
