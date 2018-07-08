x = [int(i) for i in input().split()]


def minimalProizv(x):
    q = x.index(max(x))
    qa = x.pop(q)
    c = x.index(max(x))
    ca = x.pop(c)
    z = x.index(min(x))
    za = x.pop(z)
    v = x.index(min(x))
    va = x.pop(v)

    if qa * ca > za * va:
        print(ca, qa)
    else:
        print(za, va)
    return


minimalProizv(x)
