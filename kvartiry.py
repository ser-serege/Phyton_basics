x = int(input('Первая квартира номер = '))
y = int(input('Вторая квартира номер = '))
a = int(input('Количество квартир в подезде = '))

start1 = 1
finish1 = int(start1) + a - 1

if x > finish1:
    z = x // a
    start = z * finish1 + start1
    finish = start + a
    if start <= x < finish and start < y <= finish:
        print('YES, обе квартиры в одном подъезде')
    else:
        print('NO, квартиры в разных подъездах')
elif x < finish1:
    if start1 <= x < finish1 and start1 < y <= finish1:
        print('YES, обе квартиры в одном подъезде')
    else:
        print('NO')
else:
    print('NO, квартиры в разных подъездах')
