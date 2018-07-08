from math import floor, ceil

x = float(input())

if x - floor(x) >= 0.5:
    print(ceil(x))
else:
    print(floor(x))
