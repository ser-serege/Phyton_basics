a = input()

b = []

for i in range(0, len(a), 4):
    b += a[i]
print(*b)
