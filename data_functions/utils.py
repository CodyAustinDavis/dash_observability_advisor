def safe_round(value, decimals=0):
    if value is None:
        return 0
    else:
        return round(value, decimals)