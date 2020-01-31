'''You are given an array of integers. You should find the sum of the integers with even indexes (0th, 2nd, 4th...). Then multiply this summed number and the final element of the array together. Don't forget that the first element has an index of 0.

For an empty array, the result will always be 0 (zero).

Input: A list of integers.

Output: The number as an integer.

Example:

checkio([0, 1, 2, 3, 4, 5]) == 30
checkio([1, 3, 5]) == 30
checkio([6]) == 36
checkio([]) == 0'''

def checkio(array):
    """
        sums even-indexes elements and multiply at the last
    """
    b = 0
    if len(array) > 1:
        for i in array[::2]:
            b += i
        b *= array[-1]
        return b
    if len(array) == 1:
        return array[0]*array[0]
    else:
        return 0
