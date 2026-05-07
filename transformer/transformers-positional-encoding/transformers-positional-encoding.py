import numpy as np

def positional_encoding(seq_length: int, d_model: int) -> np.ndarray:
    """
    Generate sinusoidal positional encodings.
    """
    # build a empty matrix of shape seq_length and d_model
    pe = np.empty((seq_length, d_model))
    #iterate through the matrix, fo each position sin/cos(current_idx/10000 pow(i/d))
    div_term = np.power(10000, np.arange(0, d_model, 2)/d_model)
    pos = np.arange(seq_length)[:, None]
    pe[:, 0::2] = np.sin(pos/div_term)
    pe[:, 1::2] = np.cos(pos/div_term)
    return pe