a = int(input())
b = int(input())
c = int(input())
d = int(input())

if 0 < a < 9 and 0 < b < 9 and 0 < c < 9 and 0 < d < 9:

    if a == c and b + 1 == d:
        print('YES')
    elif a == c and b - 1 == d:
        print('YES')
    elif a + 1 == c and b - 1 == d:
        print('YES')
    elif a + 1 == c and b + 1 == d:
        print('YES')
    elif a - 1 == c and b + 1 == d:
        print('YES')
    elif a - 1 == c and b - 1 == d:
        print('YES')
    elif a + 1 == c and b == d:
        print('YES')
    elif a - 1 == c and b == d:
        print('YES')
    else:
        print('NO')
else:
    print('Числа должны быть от 1 до 8')
