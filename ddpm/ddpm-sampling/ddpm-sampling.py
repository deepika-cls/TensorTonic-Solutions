import numpy as np

def ddpm_sample(
    model_predict: callable,
    shape: tuple,
    betas: np.ndarray,
    T: int
) -> np.ndarray:
    """
    Generate a sample using DDPM.
    """
    # get signal ratio and cumulative signal ratio from betas
    alphas      = 1.0 - betas                           # α_t = 1 − β_t
    alpha_bars  = np.cumprod(alphas) 
    # pure noise at timestep t of given shape
    x = np.random.randn(*shape).astype(np.float32)

    # for loop in reverse order from T till 1 
    for t in reversed(range(1, T)):
        alpha_t     = alphas[t]
        alpha_bar_t = alpha_bars[t]
        beta_t      = betas[t]
        eps_pred = model_predict(x, t)
        # denoising mean coefficient
        coeff = beta_t / np.sqrt(1 - alpha_bar_t)
        mean  = (1.0 / np.sqrt(alpha_t)) * (x - coeff * eps_pred)
        # add stochastic noise for all steps except the last
        if t == 1:
            x = mean
        else:
            sigma = np.sqrt(beta_t)                     # σ_t = √β_t
            z     = np.random.randn(*shape).astype(np.float32)
            x     = mean + sigma * z     
    return x
