'''
So this mission is the easiest one. Write a function that will receive 2 numbers as input and it should return the multiplication of these 2 numbers.
Input: Two arguments. Both are int
Output: Int.
Example
mult_two(2, 3) == 6
mult_two(1, 0) == 0
'''
def mult_two(a, b):
    c = 0
    for i in range(b):
        if b == 0:
            a = 0
        else: c += a
        
    return c
