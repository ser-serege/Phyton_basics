
A = int(input())
B = int(input())
N = int(input())

C = A * N
D = B * N

if A < 1001 and B < 1001 and N < 1001:
    if D > 99:
       K = D // 100
       D = str(D)
       L = (D[-2]+ D[-1])
       C = A * N + K
       print(C, L)
    else:
       print(int(C, D))
else:
    print('Problems')
