x = list(map(int, input().split()))

b = x.index(max(x))
c = x.index(min(x))

maxim = max(x)
minim = min(x)

for i in range(len(x)):
    if x[i] == maxim:
        x[i] = minim
    elif x[i] == minim:
        x[i] = maxim

print(' '.join(map(str, x)))
