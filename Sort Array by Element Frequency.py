'''Sort the given iterable so that its elements end up in the decreasing frequency order, that is, the number of times they appear in elements. If two elements have the same frequency, they should end up in the same order as the first appearance in the iterable.

Input: Iterable

Output: Iterable

Example:

frequency_sort([4, 6, 2, 2, 6, 4, 4, 4]) == [4, 4, 4, 4, 6, 6, 2, 2]
frequency_sort(['bob', 'bob', 'carl', 'alex', 'bob']) == ['bob', 'bob', 'bob', 'carl', 'alex']'''

def frequency_sort(items):
    count = [items.count(i) for i in items]
    
    d = dict([i for i in zip(items, count)])
    dd = list(d.items())
    dd.sort(key=lambda i: i[1], reverse=True)
    fin = [(str(i[0])) for i in dd]
    
    a = []
    for i in range(len(dd)):
        for k in range(dd[i][1]):
            a.append(dd[i][0])
    return a
