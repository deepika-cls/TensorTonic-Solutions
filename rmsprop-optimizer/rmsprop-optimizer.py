import numpy as np

def rmsprop_step(w, g, s, lr=0.001, beta=0.9, eps=1e-8):
    """
    Perform one RMSProp update step.
    """
    w = np.asarray(w)
    g = np.asarray(g)
    s = np.asarray(s)
    s_curr = beta * s + (1 - beta) * (g ** 2)
    w_curr = w - (lr / np.sqrt(s_curr + eps)) * g
    return (w_curr, s_curr)