"""Unit tests for RM code module."""

import unittest
from blincodes.matrix import Matrix
from blincodes.codes import rm


class RMCodesTestCase(unittest.TestCase):
    """Test to working with Reed--Muller codes."""

    def test_rm_generator(self):
        """Test evaluation of Reed--Muller codes generator matrix."""
        rm14_values = [
            0b1111111111111111,
            0b0000000011111111,
            0b0000111100001111,
            0b0011001100110011,
            0b0101010101010101,
        ]
        self.assertEqual(rm.generator(1, 4), Matrix(rm14_values, 16))
        self.assertTrue(
            (rm.generator(3, 8) * rm.generator(4, 8).transpose()).is_zero())

    def test_rm_parity_check(self):
        """Test evaluation of Reed--Muller codes parity-check matrix."""
        self.assertTrue(
            (rm.generator(3, 8) * rm.parity_check(4, 8).transpose()).is_zero())
        self.assertTrue(
            (rm.generator(1, 5) * rm.parity_check(3, 5).transpose()).is_zero())


if __name__ == "__main__":
    unittest.main()
