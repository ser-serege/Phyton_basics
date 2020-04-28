x = int(input())
y = int(input())

if 0 < x < 1001 and 0 < y < 1001:
    if x > y:
        print(x)
    else:
        print(y)
else:
    print('Число должно быть от 1 до 1000')
