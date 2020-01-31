'''
train
Let's teach the Robots to distinguish words and numbers.
You are given a string with words and numbers separated by whitespaces (one space). The words contains only letters. You should check if the string contains three words in succession. For example, the string "start 5 one two three 7 end" contains three words in succession.
Input: A string with words.
Output: The answer as a boolean.
Example:
checkio("Hello World hello") == True
checkio("He is 123 man") == False
checkio("1 2 3 4") == False
checkio("bla bla bla bla") == True
checkio("Hi") == False'''

def checkio(words: str) -> bool:
    words_list = words.split(' ')
    licznik = 0
    if len(words_list) < 3:
        return False
    for elements in words_list:
        if elements.isalpha():
            licznik += 1
            if licznik == 3:
                return True
        else:
            licznik = 0
    return False
    return True or False
