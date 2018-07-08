x = int(input())
y = int(input())
z = int(input())

p = (x + y + z) / 2
s = (p*(p-x)*(p-y)*(p-z))**(1/2)

print('{0:.6f}'.format(s))
