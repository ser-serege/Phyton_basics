x = str(input())
pos = x.find('f')
pos2 = x.rfind('f')

if pos == -1 and pos2 == -1:
    print()
elif pos2 != -1 and pos != pos2:
    print(pos, pos2)
elif pos == pos2 and pos2 != -1:
    print(pos)
