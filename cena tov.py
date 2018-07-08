from decimal import Decimal
from math import floor

x = Decimal(input())

a = floor(x)
b = x - a
print(a, round(b*100))
