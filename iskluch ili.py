x = int(input())
y = int(input())


def xor(x, y):
    if x != y and (x == 1 or y == 1):
        print('1')
    else:
        print('0')
        return
xor(x, y)
