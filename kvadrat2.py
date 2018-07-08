from math import sqrt

a = float(input())
b = float(input())
c = float(input())

D = b**2 - 4 * a * c
if D > 0:
    x2 = ((-1 * b) + sqrt(D)) / (2 * a)
    x1 = ((-1 * b) - sqrt(D)) / (2 * a)
    print('2', x1, x2)
elif D == 0:
    x = ((-1 * b) - sqrt(D)) / (2 * a)
    print('1', x)
elif a == 0 and b == 0 and c == 0:
        print('3')
else:
    print('0')
