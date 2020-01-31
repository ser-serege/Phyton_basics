'''In this mission your task is to determine the popularity of certain words in the tex
At the input of your function are given 2 arguments: the text and the array of words the popularity of which you need to determine.
When solving this task pay attention to the following points:
The words should be sought in all registers. This means that if you need to find a word "one" then words like "one", "One", "oNe", "ONE" etc. will do.
The search words are always indicated in the lowercase.
If the word wasn’t found even once, it has to be returned in the dictionary with 0 (zero) value.
Input: The text and the search words array.
Output: The dictionary where the search words are the keys and values are the number of times when those words are occurring in a given text.
Example
popular_words('''
When I was One
I had just begun
When I was Two
I was nearly new
''', ['i', 'was', 'three', 'near']) == {
    'i': 4,
    'was': 3,
    'three': 0,
    'near': 0}
'''
def popular_words(text: str, words: list) -> dict:
    lis = [i.lower() for i in text.split()]
    dic, pic = [], []
    for i in lis:
        dic.append(i)
        pic.append(lis.count(i))

    dictt = dict(zip(dic, pic))    
    dictt = dict(filter(lambda x: x[0] in words, dictt.items()))
    w = list(dictt.keys())
    to_add = [i for i in words if i not in w]
    for i in to_add:
        dictt.update({i: 0})
