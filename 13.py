n = int(input())
summa = 0
len = 0

while n != 0:
    summa += n
    len += 1
    n = int(input())
print(summa // len)
