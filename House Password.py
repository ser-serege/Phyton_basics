'''House Password
Input: A password as a string.
Output: Is the password safe or not as a boolean or any data type that can be converted and processed as a boolean. In the results you will see the converted results.
Example:
checkio('A1213pokl') == False
checkio('bAse730onE') == True
checkio('asasasasasasasaas') == False
checkio('QWERTYqwerty') == False
checkio('123456123456') == False
checkio('QwErTy911poqqqq') == True'''

def checkio(data: str) -> bool:

    if len(data)>=10:
        if data.islower() + data.isupper() + data.isalpha() + data.isnumeric() == 0:
            return True
    else: 
        return False
