x = str(input())

b = x.find('h')
e = x.rfind('h')

print(x[:b] + x[e + 1:])


