**Simple Calculator**
======================

A modular calculator built in Python, providing basic arithmetic operations.

**Description**
---------------

This project implements a simple calculator with add, subtract, multiply, and divide operations. The calculator is designed with a modular architecture, separating operations, utility functions, and testing into distinct files.

**Installation/Setup**
----------------------

To use the calculator, simply clone the repository and navigate to the project directory.

```bash
git clone https://github.com/your-username/simple_calculator.git
cd simple_calculator
```

**Running the Calculator**
---------------------------

To run the calculator, execute the `main.py` file using Python.

```bash
python main.py
```

This will launch a simple command-line interface, allowing you to interact with the calculator.

**Running Tests**
-----------------

To run the unit tests, execute the `test_calculator.py` file using Python.

```bash
python test_calculator.py
```

This will run the tests for each operation, ensuring the calculator functions correctly.

**File Description**
--------------------

### calculator.py

Contains the `Calculator` class, which provides methods for basic arithmetic operations:

* `add(num1, num2)`: Returns the sum of two numbers.
* `subtract(num1, num2)`: Returns the difference of two numbers.
* `multiply(num1, num2)`: Returns the product of two numbers.
* `divide(num1, num2)`: Returns the quotient of two numbers.

### utils.py

Provides utility functions for input validation and error handling:

* `is_number(input)`: Checks if the input is a valid number.
* `handle_division_by_zero()`: Handles division by zero errors.

### main.py

Serves as the entry point for the calculator, creating an instance of the `Calculator` class and providing a simple command-line interface.

### test_calculator.py

Contains unit tests for the calculator operations using the `unittest` framework. Tests each operation with various inputs and edge cases.