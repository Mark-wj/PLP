def calculator(a, b, operation):
    if operation == "+":
        final = a + b
    elif operation == "-":
        final = a - b
    elif operation == "*":
        final = a * b
    elif operation == "/":
        if b == 0:
            return "Error: Division by zero"
        final = a / b
    else:
        return "Invalid operation"
    return final

a = int(input("Enter first number: "))
b = int(input("Enter second number: "))
operation = input("Enter your desired operation (+, -, *, /): ")
print("Result:", calculator(a, b, operation))
