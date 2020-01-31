'''
There are four substring missions that were born all in one day and you shouldn’t need more than one day to solve them. All of these missions can be simply solved by brute force, but is it always the best way to go? (you might not have access to all of those missions yet, but they are going to be available with more opened islands on the map).
This mission is the first one of the series. Here you should find the length of the longest substring that consists of the same letter. For example, line "aaabbcaaaa" contains four substrings with the same letters "aaa", "bb","c" and "aaaa". The last substring is the longest one, which makes it the answer.
Input: String.
Output: Int.
Example:
long_repeat('sdsffffse') == 4
long_repeat('ddvvrwwwrggg') == 3
'''
def long_repeat(line: str) -> int:
    if len(line)>0:
        c = []
        b = [line[0]]
        for i in range(1,len(line)):
            if line[i-1] == line[i]:
                b.append(line[i])
                c.append(len(b))
            else:
                b = [line[i]]
        try:
            return max(c)
        except: 
            return 1
    else:
        return 0
