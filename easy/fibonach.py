n = int(input())
x = 1
x1 = 1

for i in range(1,n-1):
    x = x + x1
    x1 = x
print(x)
