a = int(input())
n = int(input())

def power(a,n):
    if a == 0: return 0
    if n < 0:
        a= 1.0/a
        n= -n
    res = 1
    while n > 0:
        res = res * a
        n = n-1
    return res
