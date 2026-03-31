import numpy as np

def rnn_step_forward(x_t, h_prev, Wx, Wh, b):
    """
    Returns: h_t of shape (H,)
    """
    h_t = np.dot(x_t, Wx) + np.dot(h_prev, Wh) + b
    return np.tanh(h_t)
