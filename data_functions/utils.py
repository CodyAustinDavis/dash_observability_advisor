def safe_round(value, decimals=0):
    if value is None:
        return 0
    else:
        return round(value, decimals)
    

def safe_divide(numerator, denominator):
    # Check if either value is None and return None if so
    if numerator is None or denominator is None:
        return None
    
    # Check if the denominator is zero and return None or another specific value
    if denominator == 0:
        return None  # Or return a specific value that indicates an undefined result
    
    # Perform division if it is safe
    return numerator / denominator


def safe_add(a, b):
    # Replace None with 0
    a = 0 if a is None else a
    b = 0 if b is None else b
    
    # Perform addition
    return a + b