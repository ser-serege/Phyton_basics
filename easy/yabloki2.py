K = int(input('Количество школьников = '))
N = int(input('Количество яблок = '))

A2S = int(N % K)

c = 'яблоку'
d = 'яблока'
e = 'яблок'

if A2S == int(1 or 11 or 21):
    f = c
elif A2S == int(2 or 3 or 4 or 22 or 24):
    f = d
else:
    f = e

if N < 1001:
    print('В корзине остается', A2S, f)
else:
    print(' Яблок столько нет ! ')
    
