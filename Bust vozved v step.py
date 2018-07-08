a, n = int(input()), int(input())


def power(a, n):
    if n > 0 and n % 2 == 0:
        return a * a * power(a, ((n - 1) / 2))
    elif n > 0 and n % 2 != 0:
        return a * power(a, n - 1)
    if n == 0:
        return 1
    else:
        return 1

print(power(a, n))
