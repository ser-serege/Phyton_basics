n = int(input())
b = 0
m = 1
while n != 0:
    b += 1
    n = n - 1

for k in range(b):
    m *= 10

for l in range(m-1, (m//10)-1, -2):
    print(l, end=' ')
