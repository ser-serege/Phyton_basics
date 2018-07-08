from math import sqrt
x1 = int(input())
y1 = int(input())
x2 = int(input())
y2 = int(input())
x3 = int(input())
y3 = int(input())


def Ptreug(x1, y1, x2, y2, x3, y3):

    m = sqrt((x1 - x2)**2 + (y1 - y2)**2)
    n = sqrt((x1 - x3)**2 + (y1 - y3)**2)
    b = sqrt((x2 - x3)**2 + (y2 - y3)**2)
    p = m + n + b
    return p


print(Ptreug(x1, y1, x2, y2, x3, y3))
