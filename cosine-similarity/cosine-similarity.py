import numpy as np

def cosine_similarity(a, b):
    """
    Compute cosine similarity between two 1D NumPy arrays.
    Returns: float in [-1, 1]
    """
    dot_prod = np.dot(a, b)
    norm_prod = np.linalg.norm(a) * np.linalg.norm(b)
    return  0 if norm_prod== 0 else dot_prod / norm_prod
    