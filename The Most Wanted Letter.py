'''You are given a text, which contains different english letters and punctuation symbols. You should find the most frequent letter in the text. The letter returned must be in lower case.
While checking for the most wanted letter, casing does not matter, so for the purpose of your search, "A" == "a". Make sure you do not count punctuation symbols, digits and whitespaces, only letters.

If you have two or more letters with the same frequency, then return the letter which comes first in the latin alphabet. For example -- "one" contains "o", "n", "e" only once for each, thus we choose "e".

Input: A text for analysis as a string.

Output: The most frequent letter in lower case as a string.

Example:

checkio("Hello World!") == "l"
checkio("How do you do?") == "o"
checkio("One") == "e"
checkio("Oops!") == "o"
checkio("AAaooo!!!!") == "a"
checkio("abe") == "a"'''


def checkio(text: str) -> str:
    dic = {}
    a = []
    for  i in text:
        if i not in ('!', '.', ',', '?', ' ', '1', '2', '3', '4', '5', '6', '7', '8', '9', '0', '-', '_', '+'):
                a.append(i.lower())
    text = ''.join(a)
    
    for i in text:
        dic.update({i : text.count(i)})
    
    ddd = list(filter(lambda x: x[1] == max(list(dic.values())), dic.items()))
    
    if len(ddd)>1:
        a = []
        for i in range(len(ddd)):
            a.append(ddd[i][0])
        return min(a)
    else:
        return max(dic, key=dic.get)
