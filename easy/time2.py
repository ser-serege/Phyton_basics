
N = int(input())

if N < int(10**7):
    Hours = N // 60
    Minutes = N % 60

    if Hours < 24:
        print(Hours, Minutes)

    else:
        Days = N // (60 * 24)
        Hours = N % int(Days)
        Minutes = N % 60
        print(Days, Hours, Minutes)
else:
    print('Противоречит условию задачи')
