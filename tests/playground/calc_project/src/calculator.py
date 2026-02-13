"""Simple calculator module for testing code understanding."""

class Calculator:
    """A basic calculator with history tracking."""
    
    def __init__(self):
        self.history = []
        self._last_result = 0
    
    def add(self, a: float, b: float) -> float:
        result = a + b
        self.history.append(f"{a} + {b} = {result}")
        self._last_result = result
        return result
    
    def subtract(self, a: float, b: float) -> float:
        result = a - b
        self.history.append(f"{a} - {b} = {result}")
        self._last_result = result
        return result
    
    def multiply(self, a: float, b: float) -> float:
        result = a * b
        self.history.append(f"{a} * {b} = {result}")
        self._last_result = result
        return result
    
    def divide(self, a: float, b: float) -> float:
        if b == 0:
            raise ValueError("Cannot divide by zero")
        result = a / b
        self.history.append(f"{a} / {b} = {result}")
        self._last_result = result
        return result
    
    def power(self, base: float, exp: float) -> float:
        result = base ** exp
        self.history.append(f"{base} ^ {exp} = {result}")
        self._last_result = result
        return result
    
    def get_history(self) -> list:
        return self.history.copy()
    
    def clear_history(self):
        self.history.clear()
        self._last_result = 0
    
    @property
    def last_result(self) -> float:
        return self._last_result


def factorial(n: int) -> int:
    """Calculate factorial of n."""
    if n < 0:
        raise ValueError("Factorial undefined for negative numbers")
    if n <= 1:
        return 1
    return n * factorial(n - 1)


def fibonacci(n: int) -> list:
    """Return first n Fibonacci numbers."""
    if n <= 0:
        return []
    if n == 1:
        return [0]
    fib = [0, 1]
    for i in range(2, n):
        fib.append(fib[i-1] + fib[i-2])
    return fib
