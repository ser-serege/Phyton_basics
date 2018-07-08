a = list(map(int, input().split()))


def naim_pol_chis(a):
    k = []
    for i in range(0, len(a)):
        if abs(a[i]) <= 1000:
            if a[i] > 0:
                k.append(a[i])
    b = min(k)
    return b


print(naim_pol_chis(a))
