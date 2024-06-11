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


## Handle Local Updates to Store (do NOT clear store)
# Accumulate changes for new records
def group_changes_by_row(changes, row_index_col = 'rowIndex'):
    try: 

        if changes:

            grouped_changes = {}
            for change in changes:
                row_index = change[row_index_col]
                if row_index not in grouped_changes:
                    grouped_changes[row_index] = change.copy()
                else:
                    for k, v in change.items():
                        if v:
                            grouped_changes[row_index][k] = v

            return list(grouped_changes.values())

        else:
            return []

    except Exception as e:
        raise(f"ERROR Grouping changes {changes} \n With error: {str(e)}")
    

# Helper function to mark changed rows
def mark_changed_rows(data, changes, row_id:str = 'rowIndex'):

    for row in data:
        try: 
            row['isChanged'] = 1 if any(row[row_id] == change[row_id] for change in changes) else 0

        except Exception as e:
            raise(Exception(f"ERROR marking changed cells in row \n {row} with changes: {changes} with error {str(e)}"))

    return data