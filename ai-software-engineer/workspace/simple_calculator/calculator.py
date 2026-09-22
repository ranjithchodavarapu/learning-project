from utils import is_number, handle_division_by_zero

class Calculator:
    """
    A simple calculator class with methods for basic arithmetic operations.
    """

    def add(self, num1, num2):
        """
        Adds two numbers.

        Args:
            num1 (float): The first number.
            num2 (float): The second number.

        Returns:
            float: The sum of num1 and num2.
        """
        if not is_number(num1) or not is_number(num2):
            raise ValueError("Both inputs must be numbers")
        return num1 + num2

    def subtract(self, num1, num2):
        """
        Subtracts num2 from num1.

        Args:
            num1 (float): The first number.
            num2 (float): The second number.

        Returns:
            float: The difference of num1 and num2.
        """
        if not is_number(num1) or not is_number(num2):
            raise ValueError("Both inputs must be numbers")
        return num1 - num2

    def multiply(self, num1, num2):
        """
        Multiplies two numbers.

        Args:
            num1 (float): The first number.
            num2 (float): The second number.

        Returns:
            float: The product of num1 and num2.
        """
        if not is_number(num1) or not is_number(num2):
            raise ValueError("Both inputs must be numbers")
        return num1 * num2

    def divide(self, num1, num2):
        """
        Divides num1 by num2.

        Args:
            num1 (float): The dividend.
            num2 (float): The divisor.

        Returns:
            float: The quotient of num1 and num2.

        Raises:
            ZeroDivisionError: If num2 is zero.
        """
        if not is_number(num1) or not is_number(num2):
            raise ValueError("Both inputs must be numbers")
        if num2 == 0:
            raise ZeroDivisionError("Cannot divide by zero")
        return num1 / num2