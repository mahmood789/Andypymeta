def test_import():
    import andypymeta
    assert hasattr(andypymeta, "__version__")

def test_numpy_available():
    import numpy as np
    import pandas as pd
    assert np.__version__ and pd.__version__
