n = int(input())
summa = 0

while n != 0:
    if n % 2 == 0:
        summa += n
    n = int(input())
print(summa)
