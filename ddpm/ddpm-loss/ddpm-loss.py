import numpy as np

def compute_ddpm_loss(
    model_predict: callable,
    x_0: np.ndarray,
    betas: np.ndarray,
    T: int
) -> float:
    """
    Compute DDPM training loss for a batch of images.
    """
    #calculate alphas and alpha_bars from betas
    alphas = 1.0 - betas
    alpha_bars = np.cumprod(alphas)
    # given timestep lets compute forward ddpm which will give x_t
    #sample t
    epsilon = np.random.randn(*x_0.shape)
    t = np.random.randint(0, T, size=x_0.shape[0])
    alpha_bar_t = np.expand_dims(alpha_bars[t], axis=(1,2,3))
    x_t = (
        np.sqrt(alpha_bar_t) * x_0 +
        np.sqrt(1 - alpha_bar_t) * epsilon
    )
    
    epsilon_pred = model_predict(x_t, t)
    #compute mse loss
    loss = np.mean((epsilon - epsilon_pred) ** 2)
    return loss
