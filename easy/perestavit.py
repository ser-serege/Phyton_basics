x = [int(i) for i in input().split()]
x1 = 0
x2 = 0

for i in range(1, len(x), 2):
    x1 = x[i]
    x2 = x[i-1]
    x[i] = x2
    x[i-1] = x1

print(' '.join(map(str, x)))
