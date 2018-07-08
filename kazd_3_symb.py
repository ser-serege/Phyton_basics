x = input()

b = x[::3]
c = b
v = len(c)
while v != 0:
    c = b[v-1]
    x = x.replace(c, '', 1)
    v = v - 1
print(x)
