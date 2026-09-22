import calculator
import utils

def get_number(prompt):
    """Get a number from the user."""
    while True:
        try:
            return float(input(prompt))
        except ValueError:
            print("Invalid input. Please enter a number.")

def main():
    """Run the calculator."""
    calc = calculator.Calculator()

    while True:
        print("\nOptions:")
        print("1. Add")
        print("2. Subtract")
        print("3. Multiply")
        print("4. Divide")
        print("5. Quit")

        choice = input("Choose an option: ")

        if choice == "5":
            break

        if choice not in "1234":
            print("Invalid choice. Please choose a valid option.")
            continue

        num1 = get_number("Enter the first number: ")
        num2 = get_number("Enter the second number: ")

        if choice == "1":
            result = calc.add(num1, num2)
            print(f"{num1} + {num2} = {result}")
        elif choice == "2":
            result = calc.subtract(num1, num2)
            print(f"{num1} - {num2} = {result}")
        elif choice == "3":
            result = calc.multiply(num1, num2)
            print(f"{num1} * {num2} = {result}")
        elif choice == "4":
            try:
                result = calc.divide(num1, num2)
                print(f"{num1} / {num2} = {result}")
            except ZeroDivisionError:
                print("Error: Division by zero is not allowed.")

if __name__ == "__main__":
    main()