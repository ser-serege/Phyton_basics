x = int(input())

z = 0

while x != 0:
    if x > z:
        z = x
        len += 1
    else:
        z = x
    x = int(input())
print(len)
