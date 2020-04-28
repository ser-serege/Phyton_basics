a = int(input())
b = int(input())
n = int(input())

if 0 < a < 10001 and 0 < b < 100 and 0 < n < 10001:
    z = a * n
    q = b * n
    if q >= 100:
        z = z + int(str(q)[0])
        q = int(str(q)[1:3])
    print(z, q)
