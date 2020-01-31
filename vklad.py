'''You are given the current stock prices. You have to find out which stocks cost more.
Input: The dictionary where the market identifier code is a key and the value is a stock price.
Output: The market identifier code (ticker symbol) as a string.
Example:
best_stock({"CAC": 10.0, "ATX": 390.2, "WIG": 1.2}) == "ATX"
best_stock({"CAC": 91.1, "ATX": 1.01, "TASI": 120.9}) == "TASI"'''

def best_stock(a):
    return max(a, key=a.get)
