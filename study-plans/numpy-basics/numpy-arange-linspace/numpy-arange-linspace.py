import numpy as np

def create_sequence(start, stop, param, kind):
    """
    Returns: 1D ndarray of float64 values
    """
    match kind:
        case 'arange':
            return np.arange(start, stop, param, dtype=np.float64)
        case 'linspace':
            return np.linspace(start, stop, param, dtype=np.float64)
        case _:
            raise ValueError(f"Unknown kind: {kind!r}")
    
