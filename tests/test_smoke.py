def test_import():
    import andypymeta
    assert hasattr(andypymeta, "__version__")

def test_numpy_available():
    import numpy as np
    import pandas as pd
    assert np.__version__ and pd.__version__

def test_continuity_correction():
    """Test the syntax error fix in core.py"""
    from andypymeta.core import process_continuity_correction
    
    # Test with zeros - should add cc_value
    result = process_continuity_correction(0, 0, 1, 1, cc_value=0.5)
    assert result == (0.5, 0.5, 1, 1)
    
    # Test with non-zeros - should not change
    result = process_continuity_correction(1, 2, 3, 4, cc_value=0.5)
    assert result == (1, 2, 3, 4)
    
    # Test cc_value is float
    result = process_continuity_correction(0, 0, 0, 0, cc_value=1)
    assert result == (1.0, 1.0, 1.0, 1.0)
    assert isinstance(result[0], float)