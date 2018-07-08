a = list(map(int, input().split()))
k = int(input())


def del_k_ind(a, k):
    a.pop(k)
    return a


b = ' '.join(map(str, del_k_ind(a, k)))
print(b)
