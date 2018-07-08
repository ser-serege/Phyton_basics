a, b = int(input()), int(input())


def sum(a, b):
    if b >= 0:
        return a + sum(1, b - 1)
    else:
        return 0


print(sum(a, b))
