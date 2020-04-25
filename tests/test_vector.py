"""Unit Tests for module vector."""
import unittest
from blincodes import vector


class InitVectorTestCase(unittest.TestCase):
    """Testing initialisation of Vector object."""

    def test_get_int_value_default(self):
        """Test to get value and represent as integer of default Vector."""
        vec = vector.Vector()
        self.assertEqual(int(vec), vec.value)

    def test_get_int_value(self):
        """Test to get value and represent as integer of Vector."""
        vec = vector.Vector(0b0100011011101, 13)
        self.assertEqual(int(vec), vec.value)

    def test_init_default(self):
        """Test init by default values."""
        vec = vector.Vector()
        self.assertEqual(int(vec), 0)

    def test_init_by_integer(self):
        """Test init by integer."""
        vec = vector.Vector(0b001101, 6)
        self.assertEqual(int(vec), 0b001101)

    def test_make_vector_from_string(self):
        """Test make vector from string."""
        vec = vector.from_string(
            '*1&11-0110|*1-',
            onefillers='&|',
            zerofillers='*-')
        self.assertEqual(int(vec), 0b01111001101010)

    def test_make_vector_from_iterable(self):
        """Test make vector from any iterable type."""
        vec = vector.from_iterable(
            ('*', 1, '&', 1, 1, '-', '0', '1', 1, 0, '|*1-'),
            onefillers='&|',
            zerofillers='*-')
        self.assertEqual(int(vec), 0b01111001101010)

    def test_make_vector_from_support(self):
        """Test make vector from support."""
        vec = vector.from_support(14, (1, 2, 3, 4, 7, 8, 12))
        self.assertEqual(int(vec), 0b01111001100010)

    def test_make_vector_from_support_supplement(self):
        """Test make vector from support supplement."""
        vec = vector.from_support_supplement(14, (0, 5, 6, 9, 10, 11, 13))
        self.assertEqual(int(vec), 0b01111001100010)


class TestRepesentVectorAsStringTestCase(unittest.TestCase):
    """Testing to representation Vector object as string."""

    def test_default_to_str_empty(self):
        """Test representation as string."""
        vec = vector.Vector()
        self.assertEqual(str(vec), '')

    def test_default_to_str(self):
        """Test representation as string."""
        vec = vector.Vector(0b0111100110, 10)
        self.assertEqual(str(vec), '0111100110')

    def test_function_to_str(self):
        """Test general function to_str()."""
        vec = vector.Vector(0b0111100110, 10)
        self.assertEqual(vec.to_str(), str(vec))
        vec = vector.Vector()
        self.assertEqual(vec.to_str(), str(vec))

    def test_function_to_str_with_fillers(self):
        """Test representation as string using various fillers."""
        vec = vector.Vector(0b0111100110, 10)
        self.assertEqual(vec.to_str(zerofiller='-', onefiller='$'),
                         '-$$$$--$$-')

    def test_repr_function_for_empty_vector(self):
        """Test repr function for empty Vector object."""
        self.assertEqual(repr(vector.Vector()), 'Vector(len=0, [])')

    def test_repr_function(self):
        """Test repr function for Vector object."""
        self.assertEqual(repr(vector.Vector(0b0111100110, 10)),
                         'Vector(len=10, [0111100110])')

    def test_to_latex_str(self):
        """Test function to_latex_str()."""
        vec1 = vector.Vector(0b0111100110, 10)
        vec2 = vector.Vector()
        self.assertEqual(vec1.to_latex_str(), '0&1&1&1&1&0&0&1&1&0')
        self.assertEqual(vec2.to_latex_str(), '')


class TestArithmeticOperationTestCase(unittest.TestCase):
    """Testing to make arithmetic operations the Vectors."""

    def test_add(self):
        """Test v1 + v2."""
        vec1 = vector.Vector(0b01101, 5)
        vec2 = vector.Vector()
        vec3 = vector.Vector(0b00101, 5)
        summ = vector.Vector(0b01000, 5)
        self.assertEqual(vec1 + vec3, summ)
        self.assertEqual(vec1 + vec2, vec1)
        vec1 += vec3
        self.assertEqual(vec1, summ)

    def test_mul(self):
        """Test v1 * v2."""
        vec1 = vector.Vector(0b01101, 5)
        vec2 = vector.Vector()
        vec3 = vector.Vector(0b00101, 5)
        zero = vector.Vector(0, 5)
        mul = vector.Vector(0b00101, 5)
        self.assertEqual(vec1 * vec3, mul)
        self.assertEqual(vec1 * vec2, zero)
        vec1 *= vec3
        self.assertEqual(vec1, mul)

    def test_xor(self):
        """Test v1 ^ v2."""
        vec1 = vector.Vector(0b01101, 5)
        vec2 = vector.Vector()
        vec3 = vector.Vector(0b00101, 5)
        summ = vector.Vector(0b01000, 5)
        self.assertEqual(vec1 ^ vec3, summ)
        self.assertEqual(vec1 ^ vec2, vec1)
        vec1 ^= vec3
        self.assertEqual(vec1, summ)

    def test_and(self):
        """Test v1 & v2."""
        vec1 = vector.Vector(0b01101, 5)
        vec2 = vector.Vector()
        vec3 = vector.Vector(0b00101, 5)
        zero = vector.Vector(0, 5)
        mul = vector.Vector(0b00101, 5)
        self.assertEqual(vec1 & vec3, mul)
        self.assertEqual(vec1 & vec2, zero)
        vec1 &= vec3
        self.assertEqual(vec1, mul)

    def test_or(self):
        """Test v1 | v2."""
        vec1 = vector.Vector(0b01101, 5)
        vec2 = vector.Vector()
        vec3 = vector.Vector(0b00101, 5)
        vec_or = vector.Vector(0b01101, 5)
        self.assertEqual(vec1 | vec3, vec_or)
        self.assertEqual(vec1 | vec2, vec1)
        vec1 |= vec3
        self.assertEqual(vec1, vec_or)

    def test_bitwise_not(self):
        """Test ~v."""
        vec1 = vector.Vector(0b01101, 5)
        vec2 = vector.Vector()
        vec3 = vector.Vector(0b00000, 5)
        self.assertEqual(vec1.bitwise_not().value, 0b10010)
        self.assertEqual(vec2.bitwise_not().value, 0)
        self.assertEqual(vec3.bitwise_not().value, 0b11111)
        self.assertEqual(vector.bitwise_not(vec1).value, 0b01101)
        self.assertEqual(vector.bitwise_not(vec2).value, 0)
        self.assertEqual(vector.bitwise_not(vec3).value, 0)

    def test_shifting(self):
        """Test vector shift operators."""
        vec1 = vector.Vector(0b10011, 5)
        vec2 = vector.Vector(0b01100, 5)
        vec3 = vector.Vector(0b00100, 5)
        vec4 = vector.Vector(0b00001, 5)
        vec_empty = vector.Vector()
        self.assertEqual(vec2, vec1 << 2)
        self.assertEqual(vec3, vec1 >> 2)
        vec1 <<= 2
        self.assertEqual(vec2, vec1)
        vec1 >>= 3
        self.assertEqual(vec4, vec1)
        self.assertEqual(vec_empty, vec_empty << 10)
        self.assertEqual(vec_empty, vec_empty >> 10)


class TestWorkingWithItemsTestCase(unittest.TestCase):
    """Testing to work with items of Vector object."""

    def test_getitem(self):
        """Test to get item and set item."""
        vec = vector.Vector(0b0111100110, 10)
        self.assertEqual(vec[2], 1)
        self.assertEqual(vec[0], 0)
        self.assertEqual(vec[-1], 0)
        self.assertEqual(vec[-8], 1)
        self.assertEqual(vec[-5], 0)
        self.assertEqual(int(vec[1:6:2]), 0b110)

    def test_setitem_positive_positions(self):
        """Test to set item for positive positions."""
        vec = vector.Vector(0b0111100110, 10)
        vec[0] = 1
        vec[1] = 1
        vec[2] = ''
        vec[3] = '0'
        vec[4] = '1'
        vec[5] = 0
        vec[6] = False
        vec[7] = True
        vec[8] = 'UUU'
        vec[9] = -9
        self.assertEqual(vec.value, 0b1100100111)

    def test_setitem_negative_positions(self):
        """Test to set item for negative positions."""
        vec = vector.Vector(0b0111100110, 10)
        vec[-1] = 1
        vec[-2] = ''
        vec[-3] = '0'
        vec[-4] = '1'
        vec[-5] = 0
        vec[-6] = False
        vec[-7] = True
        vec[-8] = 'UUU'
        vec[-9] = -9
        self.assertEqual(vec.value, 0b0111001001)


class EqualitiesTestCase(unittest.TestCase):
    """Testing to evaluate of various equalities."""

    def test_eq(self):
        """Test equality of vectors."""
        vec1 = vector.Vector(0b01101, 5)
        vec2 = vector.Vector()
        vec3 = vector.Vector(0b01101, 5)
        self.assertEqual(vec1, vec3)
        self.assertEqual(vec2, vec2)
        self.assertEqual(vec1, vec1)

    def test_not_eq(self):
        """Test not equality of vectors."""
        vec1 = vector.Vector(0b01101, 5)
        vec2 = vector.Vector()
        vec3 = vector.Vector(0b01001, 5)
        self.assertNotEqual(vec1, vec3)
        self.assertNotEqual(vec1, vec2)
        self.assertNotEqual(vec1, vec3)

    def test_bool(self):
        """Test comparison with None."""
        vec1 = vector.Vector(0b10011, 5)
        vec2 = vector.Vector(0b00000, 5)
        vec3 = vector.Vector()
        self.assertTrue(vec1)
        self.assertTrue(vec2)
        self.assertFalse(vec3)


class ChangeLengthTestCase(unittest.TestCase):
    """Testing to change length of a Vector object."""

    def test_get_len_empty_object(self):
        """Test to get length of empty object."""
        self.assertEqual(len(vector.Vector()), 0)

    def test_get_len(self):
        """Test to get length."""
        self.assertEqual(len(vector.Vector(0, 15)), 15)
        self.assertEqual(len(vector.Vector(0b10111011, 10)), 10)

    def test_set_size_and_resize(self):
        """Test resizing of vector."""
        vec1 = vector.Vector(0b10011, 5)
        vec2 = vector.Vector(0b011, 3)
        vec3 = vector.Vector(0b0010011, 7)
        self.assertEqual(vec2, vec1.copy().set_length(3))
        self.assertEqual(vec3, vec1.copy().set_length(7))
        self.assertEqual(vec2, vec1.copy().resize(-2))
        self.assertEqual(vec3, vec1.copy().resize(2))


class ToolFunctionsEvaluationTestCase(unittest.TestCase):
    """Testing to evaluating of various tool functions."""

    def test_make_copy(self):
        """Test make copy the Vector object."""
        vec = vector.Vector(0b0111100110, 10)
        vec_copy = vec.copy()
        self.assertEqual(vec, vec_copy)
        self.assertEqual(vector.Vector().copy(), vector.Vector())

    def test_hamming_weight(self):
        """Test evaluation ot Hamming weight."""
        vec1 = vector.Vector(0b10011, 5)
        vec2 = vector.Vector(0b01100, 5)
        vec3 = vector.Vector(0b00100, 5)
        vec4 = vector.Vector(0b00000, 5)
        vec5 = vector.Vector()
        self.assertEqual(vec1.hamming_weight, 3)
        self.assertEqual(vec2.hamming_weight, 2)
        self.assertEqual(vec3.hamming_weight, 1)
        self.assertEqual(vec4.hamming_weight, 0)
        self.assertEqual(vec5.hamming_weight, 0)
        self.assertEqual(vector.hamming_distance(vec1, vec2), 5)
        self.assertEqual(vector.hamming_distance(vec1, vec3), 4)
        self.assertEqual(vector.hamming_distance(vec3, vec4), 1)
        self.assertEqual(vector.hamming_distance(vec2, vec3), 1)
        self.assertEqual(vector.hamming_distance(vec2, vec5), 2)

    def test_support(self):
        """Test evaluation ot vector's support."""
        vec1 = vector.Vector(0b10011, 5)
        vec2 = vector.Vector(0b01100, 5)
        vec3 = vector.Vector(0b00100, 5)
        vec4 = vector.Vector(0b00000, 5)
        vec5 = vector.Vector()
        self.assertEqual(vec1.support, [0, 3, 4])
        self.assertEqual(vec2.support, [1, 2])
        self.assertEqual(vec3.support, [2])
        self.assertEqual(vec4.support, [])
        self.assertEqual(vec5.support, [])
        self.assertEqual(list(vec1.iter_support()), [0, 3, 4])
        self.assertEqual(list(vec2.iter_support()), [1, 2])
        self.assertEqual(list(vec3.iter_support()), [2])
        self.assertEqual(list(vec4.iter_support()), [])
        self.assertEqual(list(vec5.iter_support()), [])
        self.assertEqual(vec1.support_supplement, [1, 2])
        self.assertEqual(vec2.support_supplement, [0, 3, 4])
        self.assertEqual(vec3.support_supplement, [0, 1, 3, 4])
        self.assertEqual(vec4.support_supplement, [0, 1, 2, 3, 4])
        self.assertEqual(vec5.support_supplement, [])
        self.assertEqual(list(vec1.iter_support_supplement()), [1, 2])
        self.assertEqual(list(vec2.iter_support_supplement()), [0, 3, 4])
        self.assertEqual(list(vec3.iter_support_supplement()), [0, 1, 3, 4])
        self.assertEqual(list(vec4.iter_support_supplement()), [0, 1, 2, 3, 4])
        self.assertEqual(list(vec5.iter_support_supplement()), [])

    def test_concatenate(self):
        """Test concatenation of vectors."""
        vec1 = vector.Vector(0b10011, 5)
        vec2 = vector.Vector(0b00000, 5)
        vec3 = vector.Vector()
        self.assertEqual(vec1.concatenate(vec2).value, 0b1001100000)
        self.assertEqual(vec1.concatenate(vec3).value, 0b1001100000)
        self.assertEqual(vector.concatenate(vec1, vec1).value,
                         0b10011000001001100000)
        self.assertEqual(vector.concatenate(vec3, vec1).value, 0b1001100000)

    def test_scalar_product(self):
        """Test scalar product of vectors."""
        vec1 = vector.Vector(0b10011, 5)
        vec2 = vector.Vector(0b00000, 5)
        vec3 = vector.Vector()
        vec4 = vector.Vector(0b00011, 5)
        self.assertEqual(vector.scalar_product(vec1, vec1), 1)
        self.assertEqual(vector.scalar_product(vec1, vec2), 0)
        self.assertEqual(vector.scalar_product(vec1, vec4), 0)
        self.assertEqual(vector.scalar_product(vec1, vec3), 0)
        self.assertEqual(vector.scalar_product(vec3, vec3), 0)
        self.assertEqual(vector.scalar_product(vec4, vec3), 0)


if __name__ == "__main__":
    unittest.main()
