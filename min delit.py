from math import sqrt
n = int(input())


def MinDivisor(n):
    x = 2
    while n % x != 0:
        if x >= sqrt(n):
            print(n)
            return
        x += 1
    print(x)
MinDivisor(n)
