n = int(input())


def IsPrime(n):
    p = 0
    x = 1
    while x <= n:
        if n % x == 0:
            p += 1
            if p == 3:
                break
        x += 1
    return p


if IsPrime(n) > 2:
    print('NO')
else:
    print('YES')
