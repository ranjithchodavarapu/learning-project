import unittest
from calculator import Calculator
from utils import is_number, handle_division_by_zero

class TestCalculator(unittest.TestCase):
    def setUp(self):
        self.calc = Calculator()

    def test_add(self):
        self.assertEqual(self.calc.add(2, 3), 5)
        self.assertEqual(self.calc.add(-2, 3), 1)
        self.assertEqual(self.calc.add(-2, -3), -5)
        self.assertEqual(self.calc.add(0, 0), 0)

    def test_add_invalid_input(self):
        with self.assertRaises(ValueError):
            self.calc.add('a', 3)
        with self.assertRaises(ValueError):
            self.calc.add(2, 'b')

    def test_subtract(self):
        self.assertEqual(self.calc.subtract(5, 3), 2)
        self.assertEqual(self.calc.subtract(-2, 3), -5)
        self.assertEqual(self.calc.subtract(-2, -3), 1)
        self.assertEqual(self.calc.subtract(0, 0), 0)

    def test_subtract_invalid_input(self):
        with self.assertRaises(ValueError):
            self.calc.subtract('a', 3)
        with self.assertRaises(ValueError):
            self.calc.subtract(2, 'b')

    def test_multiply(self):
        self.assertEqual(self.calc.multiply(2, 3), 6)
        self.assertEqual(self.calc.multiply(-2, 3), -6)
        self.assertEqual(self.calc.multiply(-2, -3), 6)
        self.assertEqual(self.calc.multiply(0, 0), 0)

    def test_multiply_invalid_input(self):
        with self.assertRaises(ValueError):
            self.calc.multiply('a', 3)
        with self.assertRaises(ValueError):
            self.calc.multiply(2, 'b')

    def test_divide(self):
        self.assertEqual(self.calc.divide(6, 3), 2)
        self.assertEqual(self.calc.divide(-6, 3), -2)
        self.assertEqual(self.calc.divide(-6, -3), 2)
        self.assertEqual(self.calc.divide(0, 3), 0)

    def test_divide_invalid_input(self):
        with self.assertRaises(ValueError):
            self.calc.divide('a', 3)
        with self.assertRaises(ValueError):
            self.calc.divide(2, 'b')

    def test_divide_by_zero(self):
        with self.assertRaises(ZeroDivisionError):
            self.calc.divide(2, 0)

    def test_is_number(self):
        self.assertTrue(is_number(5))
        self.assertTrue(is_number(-5))
        self.assertTrue(is_number(0))
        self.assertFalse(is_number('a'))

    def test_handle_division_by_zero(self):
        with self.assertRaises(ZeroDivisionError):
            handle_division_by_zero(2, 0)

if __name__ == '__main__':
    unittest.main()