n, m = int(input()), int(input())


def ReduceFraction(n, m):
    if n < m:
        if n > 1:
            if m % n == 0:
                p = n // n
                q = m // n
                print(p, q)
            else:
                w = 1
                while w <= n // 2:
                    if n % w == 0 and m % w == 0:
                        p = n // w
                        q = m // w
                    w += 1
                print(p, q)


ReduceFraction(n, m)
