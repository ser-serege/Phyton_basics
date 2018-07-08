a = list(map(int, input().split()))
d = list(map(int, input().split()))


def add_c_on_k_ind(a, d):
    a.insert(d[0], d[1])
    return a


b = ' '.join(map(str, add_c_on_k_ind(a, d)))
print(b)
