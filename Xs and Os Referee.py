'''
Tic-Tac-Toe, sometimes also known as Xs and Os, is a game for two players (X and O) who take turns marking the spaces in a 3×3 grid. The player who succeeds in placing three respective marks in a horizontal, vertical, or diagonal rows (NW-SE and NE-SW) wins the game.
But we will not be playing this game. You will be the referee for this games results. You are given a result of a game and you must determine if the game ends in a win or a draw as well as who will be the winner. Make sure to return "X" if the X-player wins and "O" if the O-player wins. If the game is a draw, return "D".
A game's result is presented as a list of strings, where "X" and "O" are players' marks and "." is the empty cell.
Input: A game result as a list of strings (unicode).
Output: "X", "O" or "D" as a string.
Example:
checkio([
    "X.O",
    "XX.",
    "XOO"]) == "X"
checkio([
    "OO.",
    "XOX",
    "XOX"]) == "O"
checkio([
    "OOX",
    "XXO",
    "OXX"]) == "D"
'''

from typing import List

def checkio(game_result: List[str]) -> str:
    b = []
    for k in range(len(game_result)):
        for i in range(len(game_result)):
            b.append(game_result[i][k]) 
    
    C = b[:3], b[3:6], b[6:9], b[0::3], b[1::3], b[2::3], b[0::4], b[-3::-3], b[-3::-2][0:3]
    
    for i in C:
        if 'XXX' == ''.join(i):
            return 'X'
        if 'OOO' == ''.join(i):
            return 'O'
    return 'D'
    
