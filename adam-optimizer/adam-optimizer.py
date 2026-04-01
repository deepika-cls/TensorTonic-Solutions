import numpy as np

def adam_step(param, grad, m, v, t, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
    """
    One Adam optimizer update step.
    Return (param_new, m_new, v_new).
    """
    # update moment for the current timestep ()
    grad = np.asarray(grad)
    m = np.asarray(m)
    v = np.asarray(v)
    m_t = (beta1 * m) + ((1 - beta1) * grad)
    # update second momment for the current timestep
    v_t = (beta2 * v) + ((1 - beta2) * (grad ** 2))
    # bias correction
    m_t_bar = m_t / ( 1 - beta1**t)
    v_t_bar = v_t / (1 - beta2**t)
    # update param
    new_param = param - lr * (m_t_bar / (np.sqrt(v_t_bar) + eps))
    return new_param, m_t, v_t