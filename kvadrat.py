from math import sqrt

a = float(input())
b = float(input())
c = float(input())

if a != 0:
    D = b**2 - 4 * a * c
    if D > 0:
        x2 = ((-1 * b) + sqrt(D)) / (2 * a)
        x1 = ((-1 * b) - sqrt(D)) / (2 * a)
        print(x1, x2)
    elif D == 0:
        x1 = ((-1 * b) - sqrt(D)) / (2 * a)
        print(x1)
    elif D < 0:
        print()
