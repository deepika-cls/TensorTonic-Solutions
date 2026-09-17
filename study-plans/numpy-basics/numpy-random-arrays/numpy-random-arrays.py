import numpy as np

def generate_random_array(shape, kind, seed):
    """
    Returns: 2D ndarray of float64 random values
    """
    rng = np.random.default_rng(seed)
    match kind:
        case 'uniform':
            return rng.uniform(size=shape).astype(np.float64)
        case 'normal':
            return rng.normal(size=shape).astype(np.float64)
