a = input().split()

for i in range(len(a)-1):
    n = int(a[i])
    i += 1
    m = int(a[i])
    if m > n:
        n = m
        print(m, end=' ')
