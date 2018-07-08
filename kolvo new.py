a = [int(i) for i in input().split()]
pos = 1

for i in range(1, len(a)):
    x = a[i-1]
    if x != a[i]:
        pos += 1

print(pos)
