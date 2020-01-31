'''
You are given an array of numbers (floats). You should find the difference between the maximum and minimum element. Your function should be able to handle an undefined amount of arguments. For an empty argument list, the function should return 0.
Floating-point numbers are represented in computer hardware as base 2 (binary) fractions. So we should check the result with ±0.001 precision.
Think about how to work with an arbitrary number of arguments.
Input: An arbitrary number of arguments as numbers (int, float).
Output: The difference between maximum and minimum as a number (int, float).
Example:
checkio(1, 2, 3) == 2
checkio(5, -5) == 10
checkio(10.2, -2.2, 0, 1.1, 0.5) == 12.4
checkio() == 0
'''

def checkio(*args):
    try:
        return max(*args) - min(*args)
    except: 
        return 0
