"""Unit tests for codes.tools module."""

import unittest
from blincodes.matrix import Matrix
from blincodes.vector import Vector
from blincodes.codes import tools


class MakeCodesMatriciesTestCase(unittest.TestCase):
    """Testing making generator and parity check matrixes of code."""

    def setUp(self):
        """Set the test value."""
        self.rm14 = Matrix([
            0b1111111111111111,
            0b0000000011111111,
            0b0000111100001111,
            0b0011001100110011,
            0b0101010101010101,
        ], 16)

        self.rm14_add = Matrix([
            0b1111111111111111,
            0b1111111100000000,
            0b0000000011111111,
            0b0000111100001111,
            0b1111000000001111,
            0b0110011001100110,
            0b0011001100110011,
            0b1010101010101010,
            0b0101010101010101,
        ], 16)

        self.rm24 = Matrix([
            0b1111111111111111,
            0b0000000011111111,
            0b0000111100001111,
            0b0011001100110011,
            0b0101010101010101,
            0b0000000000001111,
            0b0000000011001100,
            0b0000000001010101,
            0b0000001100000011,
            0b0000010100000101,
            0b0001000100010001
        ], 16)

        self.rm24_add = Matrix([
            0b0000000001010101,
            0b1111111111111111,
            0b0000000011111111,
            0b0101010101010101,
            0b0001000100010001,
            0b0000001100000011,
            0b0000010100000101,
            0b0001000100010001,
            0b1111111111111111,
            0b1111111100000000,
            0b0000000000001111,
            0b0000000011001100,
            0b0011001100110011,
            0b0000001100000011,
            0b0000000001010101,
            0b0000000011111111,
            0b0000111100001111,
            0b1111000000001111,
            0b0000000011111111,
            0b0110011001100110,
            0b0000010100000101,
            0b0011001100110011,
            0b1010101010101010,
            0b0000000000001111,
            0b0101010101010101,
            0b1111111111111111,
            0b0000111100001111,
            0b0011001100110011,
            0b0101010101010101,
            0b0000000011001100,
            0b0000111100001111,
        ], 16)

        self.rm14_generator = Matrix([
            0b1001011001101001,
            0b0101010101010101,
            0b0011001100110011,
            0b0000111100001111,
            0b0000000011111111,
        ], 16)

        self.rm24_generator = Matrix([
            0b1000000100010111,
            0b0100000100010100,
            0b0010000100010010,
            0b0001000100010001,
            0b0000100100000110,
            0b0000010100000101,
            0b0000001100000011,
            0b0000000010010110,
            0b0000000001010101,
            0b0000000000110011,
            0b0000000000001111,
        ], 16)

    def test_make_generator(self):
        """Test evaluating of generator matrix from list of code words."""
        self.assertEqual(tools.make_generator(self.rm14),
                         self.rm14_generator)
        self.assertTrue(
            (tools.make_generator(self.rm14) * self.rm24.T).is_zero())
        self.assertEqual(tools.make_generator(self.rm24),
                         self.rm24_generator)
        self.assertTrue(
            (tools.make_generator(self.rm24) * self.rm14.T).is_zero())
        self.assertEqual(tools.make_generator(self.rm14_add),
                         self.rm14_generator)
        self.assertTrue(
            (tools.make_generator(self.rm14_add) * self.rm24.T).is_zero())
        self.assertEqual(tools.make_generator(self.rm24_add),
                         self.rm24_generator)
        self.assertTrue(
            (tools.make_generator(self.rm24_add) * self.rm14.T).is_zero())

    def test_make_parity_check(self):
        """Test evaluating of parity check matrix."""
        self.assertEqual(tools.make_parity_check(
            self.rm24).diagonal_form,
                         self.rm14_generator)
        self.assertTrue(
            (tools.make_parity_check(
                self.rm24) * self.rm24.T).is_zero())
        self.assertEqual(tools.make_parity_check(
            self.rm14).diagonal_form,
                         self.rm24_generator)
        self.assertTrue(
            (tools.make_parity_check(
                self.rm14) * self.rm14.T).is_zero())
        self.assertEqual(tools.make_parity_check(
            self.rm24_add).diagonal_form,
                         self.rm14_generator)
        self.assertTrue(
            (tools.make_parity_check(
                self.rm24_add) * self.rm24.T).is_zero())
        self.assertEqual(tools.make_parity_check(
            self.rm14_add).diagonal_form,
                         self.rm24_generator)
        self.assertTrue(
            (tools.make_parity_check(
                self.rm14_add) * self.rm14.T).is_zero())


class CodeOperationsTestCase(unittest.TestCase):
    """Testing evaluating of various operations under the codes."""

    def setUp(self):
        """Set the test value."""
        self.rm14 = Matrix([
            0b1111111111111111,
            0b0000000011111111,
            0b0000111100001111,
            0b0011001100110011,
            0b0101010101010101,
        ], 16)

        self.rm14_add = Matrix([
            0b1111111111111111,
            0b1111111100000000,
            0b0000000011111111,
            0b0000111100001111,
            0b1111000000001111,
            0b0110011001100110,
            0b0011001100110011,
            0b1010101010101010,
            0b0101010101010101,
        ], 16)

        self.rm24 = Matrix([
            0b1111111111111111,
            0b0000000011111111,
            0b0000111100001111,
            0b0011001100110011,
            0b0101010101010101,
            0b0000000000001111,
            0b0000000011001100,
            0b0000000001010101,
            0b0000001100000011,
            0b0000010100000101,
            0b0001000100010001
        ], 16)

        self.rm24_add = Matrix([
            0b0000000001010101,
            0b1111111111111111,
            0b0000000011111111,
            0b0101010101010101,
            0b0001000100010001,
            0b0000001100000011,
            0b0000010100000101,
            0b0001000100010001,
            0b1111111111111111,
            0b1111111100000000,
            0b0000000000001111,
            0b0000000011001100,
            0b0011001100110011,
            0b0000001100000011,
            0b0000000001010101,
            0b0000000011111111,
            0b0000111100001111,
            0b1111000000001111,
            0b0000000011111111,
            0b0110011001100110,
            0b0000010100000101,
            0b0011001100110011,
            0b1010101010101010,
            0b0000000000001111,
            0b0101010101010101,
            0b1111111111111111,
            0b0000111100001111,
            0b0011001100110011,
            0b0101010101010101,
            0b0000000011001100,
            0b0000111100001111,
        ], 16)

        self.rm14_generator = Matrix([
            0b1001011001101001,
            0b0101010101010101,
            0b0011001100110011,
            0b0000111100001111,
            0b0000000011111111,
        ], 16)

        self.rm24_generator = Matrix([
            0b1000000100010111,
            0b0100000100010100,
            0b0010000100010010,
            0b0001000100010001,
            0b0000100100000110,
            0b0000010100000101,
            0b0000001100000011,
            0b0000000010010110,
            0b0000000001010101,
            0b0000000000110011,
            0b0000000000001111,
        ], 16)

    def test_hadamard_product(self):
        """Test to evaluate of Hadamard product of two codes."""
        self.assertEqual(tools.hadamard_product(
            self.rm14, self.rm14_add).diagonal_form,
                         self.rm24_generator)
        self.assertTrue(
            (tools.hadamard_product(
                self.rm14, self.rm14_add) * self.rm14_generator.T).is_zero())

        self.assertTrue(tools.hadamard_product(
            self.rm24, self.rm24_add).diagonal_form.is_identity())

        self.assertEqual(
            tools.hadamard_product(
                self.rm14, self.rm24_add).orthogonal,
            Matrix([0b1111111111111111], 16))

    def test_intersection(self):
        """Test to intersect of codes."""
        self.assertEqual(tools.intersection(
            self.rm14, self.rm24_add).diagonal_form,
                         self.rm14_generator)
        self.assertEqual(tools.intersection(
            self.rm24_add, self.rm24).diagonal_form,
                         self.rm24_generator)

    def test_union(self):
        """Test to union of codes."""
        self.assertEqual(tools.union(
            self.rm14, self.rm24_add).diagonal_form,
                         self.rm24_generator)

    def test_hull(self):
        """Test to evaluate of code's hull."""
        self.assertEqual(tools.hull(self.rm24_add).diagonal_form,
                         self.rm14_generator)
        self.assertEqual(tools.hull(self.rm14_add).diagonal_form,
                         self.rm14_generator)

    def test_puncture(self):
        """Test to puncture of a code."""
        rm14_puncture = Matrix([
            0b0111011100111110,
            0b0000000000111110,
            0b0000011100001110,
            0b0011001100110010,
            0b0101010100010100,
            ], 16)
        self.assertEqual(
            tools.puncture(self.rm14_add, columns=(0, 4, 8, 9, 15)),
            rm14_puncture.diagonal_form)
        self.assertEqual(
            tools.puncture(self.rm14_add,
                           columns=(0, 4, 8, 9, 15),
                           remove_zeroes=True),
            rm14_puncture.submatrix(
                (i for i in range(16)
                 if i not in [0, 4, 8, 9, 15])).diagonal_form)

    def test_truncate(self):
        """Test to truncate of a code."""
        rm14_trunc = Matrix([
            0b0000111100001111,
            0b0000000011111111,
            ], 16)
        # print()
        # print(tools.truncate(self.rm14_add, columns=(0, 1, 2)))
        self.assertEqual(
            tools.truncate(self.rm14_add, columns=(0, 4, 8, 9, 15)),
            Matrix())
        self.assertEqual(
            tools.truncate(self.rm14_add, columns=(0, 1, 2, 3)),
            rm14_trunc)
        self.assertEqual(
            tools.truncate(self.rm14_add,
                           columns=(0, 4, 8, 9, 15),
                           remove_zeroes=True),
            Matrix())
        self.assertEqual(
            tools.truncate(self.rm14_add,
                           columns=(0, 1, 2, 3),
                           remove_zeroes=True),
            rm14_trunc.submatrix(range(4, 16)))


class CodeWordsOperationsTestCase(unittest.TestCase):
    """Testing evaluating of various operations under code's words."""

    def setUp(self):
        """Set the test value."""
        self.rm14 = Matrix([
            0b1111111111111111,
            0b0000000011111111,
            0b0000111100001111,
            0b0011001100110011,
            0b0101010101010101,
        ], 16)
        self.code_words = [
            0b0000000000000000,
            0b0101010101010101,
            0b0011001100110011,
            0b0110011001100110,
            0b0000111100001111,
            0b0101101001011010,
            0b0011110000111100,
            0b0110100101101001,
            0b0000000011111111,
            0b0101010110101010,
            0b0011001111001100,
            0b0110011010011001,
            0b0000111111110000,
            0b0101101010100101,
            0b0011110011000011,
            0b0110100110010110,
            0b1111111111111111,
            0b1010101010101010,
            0b1100110011001100,
            0b1001100110011001,
            0b1111000011110000,
            0b1010010110100101,
            0b1100001111000011,
            0b1001011010010110,
            0b1111111100000000,
            0b1010101001010101,
            0b1100110000110011,
            0b1001100101100110,
            0b1111000000001111,
            0b1010010101011010,
            0b1100001100111100,
            0b1001011001101001,
        ]

    def test_iter_code_words(self):
        """Test to iterate over code words."""
        code_words = [vec.value for vec in tools.iter_codewords(self.rm14)]
        for i in code_words:
            self.assertEqual(code_words.count(i), 1)
        self.assertEqual(len(code_words), 32)
        self.assertEqual(code_words, self.code_words)

    def test_spectrum(self):
        """Test to evaluate of spectrum."""
        spectr = {i: 0 for i in range(17)}
        spectr[0] = 1
        spectr[8] = 30
        spectr[16] = 1
        self.assertEqual(tools.spectrum(self.rm14), spectr)

    def test_encode(self):
        """Test to encode of vector."""
        self.assertEqual(
            tools.encode(self.rm14, Vector(0b11111, 5)),
            Vector(0b1001011001101001, 16))
        self.assertEqual(
            tools.encode(self.rm14, Matrix([0b11111], 5)),
            Vector(0b1001011001101001, 16))

    def test_syndrome(self):
        """Test to evaluating of syndrome."""
        self.assertEqual(
            tools.syndrome(self.rm14, Vector(0b1110000000000000, 16)),
            Vector(0b10011, 5))
        self.assertEqual(
            tools.syndrome(self.rm14, Matrix([0b1110000000000000], 16)),
            Vector(0b10011, 5))


if __name__ == "__main__":
    unittest.main()
