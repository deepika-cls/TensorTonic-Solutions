import numpy as np

def reverse_step(
    x_t: np.ndarray,
    t: int,
    epsilon_pred: np.ndarray,
    betas: np.ndarray
) -> np.ndarray:
    """
    Perform one reverse diffusion step.
    """
    alphas = 1 - betas
    alpha_bar = np.cumprod(alphas)

    noise = (1 - alphas[t])/np.sqrt(1 - alpha_bar[t]) * epsilon_pred
    mean_noise = (x_t - noise) / np.sqrt(alphas[t])
    if t > 1:
        stochastic_noise = np.sqrt(betas[t]) * np.random.randn(*x_t.shape)
        x_t_prev = mean_noise + stochastic_noise
    else:
        x_t_prev = mean_noise
    return x_t_prev
