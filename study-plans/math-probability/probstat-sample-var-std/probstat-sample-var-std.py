import numpy as np

def sample_var_std(x: list) -> dict:
    """
    Returns sample variance and standard deviation as Python floats.
    """
    variance = np.var(x,ddof=1)
    std = np.std(x,ddof=1)
    return {"variance": variance, "std_dev":std}