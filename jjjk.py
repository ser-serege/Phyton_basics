x = [int(i) for i in input().split()]

k = [x.pop()]
la = ' '.join(map(str, x))
k.append(la)

print(' '.join(map(str, k)))
