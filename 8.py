x = int(input())
y = int(input())
day = 1
while y - x > 0:
    x = x + (x * 0.1)
    day += 1
print(day)
