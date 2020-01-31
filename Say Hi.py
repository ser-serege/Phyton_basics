'''n this mission you should write a function that introduces a person with the given parameter's attributes.
Input: Two arguments. String and positive integer.
Output:String.
Example:
say_hi("Alex", 32) == "Hi. My name is Alex and I'm 32 years old"
say_hi("Frank", 68) == "Hi. My name is Frank and I'm 68 years old" '''

def say_hi(name: str, age: int) -> str:
    """
        Hi!
    """
    
    return "Hi. My name is {} and I'm {} years old".format(name, age)
