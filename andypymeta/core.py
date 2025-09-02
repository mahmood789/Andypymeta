"""Core functionality for AndyPyMeta package."""


def process_continuity_correction(a, b, c, d, cc_value=0.5):
    """
    Apply continuity correction to a 2x2 table.
    
    This function demonstrates the syntax error fix mentioned in the problem statement.
    Previously had multiple if statements on one line which was a syntax error.
    """
    # Ensure cc_value is a float
    cc_value = float(cc_value)
    
    # Fixed syntax: separate if statements instead of multiple on one line
    if a == 0:
        a += cc_value
    if b == 0:
        b += cc_value
    if c == 0:
        c += cc_value
    if d == 0:
        d += cc_value
    
    return a, b, c, d