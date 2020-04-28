
a = int(input())
b = int(input())
c = int(input())
d = int(input())
e = int(input())
f = int(input())

if a != 0 and (d * a - b * c) != 0:
    y = (a * f - c * e) // (d * a - b * c)
    x = (e - b * y) // a
print(x, y)
