def is_number(input_value):
    """
    Checks if the input value is a number.

    Args:
        input_value (str): The input value to be checked.

    Returns:
        bool: True if the input value is a number, False otherwise.
    """
    try:
        float(input_value)
        return True
    except ValueError:
        return False


def handle_division_by_zero(dividend, divisor):
    """
    Handles division by zero errors.

    Args:
        dividend (float): The dividend.
        divisor (float): The divisor.

    Returns:
        float: The result of the division if divisor is not zero, otherwise raises a ZeroDivisionError.
    """
    if divisor == 0:
        raise ZeroDivisionError("Cannot divide by zero")
    return dividend / divisor


def validate_input(input_value):
    """
    Validates the input value.

    Args:
        input_value (str): The input value to be validated.

    Returns:
        float: The validated input value as a float, or raises a ValueError if the input is invalid.
    """
    if not input_value:
        raise ValueError("Input cannot be empty")
    if not is_number(input_value):
        raise ValueError("Input must be a number")
    return float(input_value)