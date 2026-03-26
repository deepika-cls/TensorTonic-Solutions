import numpy as np

def matrix_normalization(matrix, axis=None, norm_type='l2'):
    """
    Normalize a 2D matrix along specified axis using specified norm.
    """
    matrix = np.asarray(matrix)

    if matrix.ndim != 2:
        return None  # Reject non-matrices
    
    if axis is not None and not (axis == 0 or axis == 1):
        return None  # Reject invalid 2D axes
        
    if norm_type == 'l1':
        s = np.sum(np.abs(matrix), axis=axis, keepdims=True)
    elif norm_type == 'l2':
        s = np.linalg.norm(matrix, axis=axis, keepdims=True)
    elif norm_type == 'max':
        s = np.max(np.abs(matrix), axis=axis, keepdims=True)
    else:
        return None
    return np.nan_to_num(matrix / s, nan=0.0)