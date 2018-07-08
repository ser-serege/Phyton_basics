a = list(map(int, input().split()))
b = max(a)
z = a.copy()
z.reverse()
v = z.index(b)
k = len(a) - v - 1

print(b, k)
