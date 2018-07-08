from math import floor, ceil
P = int(input())
X = int(input())
Y = int(input())
i = 1

A = X + X * (P/100)
B = Y + Y * (P/100)
F = floor(A)
G = A - F
K = B // 100
C = A + K
N = floor(C)
M = round((C - N) * 100)

while


print(N, M)
