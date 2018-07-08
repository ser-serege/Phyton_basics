
def sumpos():
    x = int(input())
    if x == 0:
        return 0
    return x + sumpos()


print(sumpos())
