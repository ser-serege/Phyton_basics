x = int(input())
max = 0
len = 1

while x != 0:
    if x > max:
        max = x
    x = int(input())
    if x == max:
        len += 1
print(len)
