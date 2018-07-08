a, n = float(input()), int(input())


def power(a, n):
    if n > 0:
        return a * power(a, n - 1)
    if n == 0:
        return 1
    else:
        return 1

print(power(a, n))
