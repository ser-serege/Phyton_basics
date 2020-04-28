a = input().split()
b = []
for i in range(1, len(a)):
    if int(a[i]) > int(a[i - 1]):
        b += a[i]
        print(' '.join(b))
    elif int(a[i]) < 0:


