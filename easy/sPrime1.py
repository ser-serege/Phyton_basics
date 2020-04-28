n = int(input())


def IsPrime(n):
   d = 2
   while n % d != 0:
       d += 1
   return d == n

if IsPrime(n) is True:
    print('YES')
else:
    print('NO')
