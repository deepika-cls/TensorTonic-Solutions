def gradient_descent_quadratic(a, b, c, x0, lr, steps):
    """
    Return final x after 'steps' iterations.
    """
    for _ in range(steps):
        grad = 2 * x0 * a + b
        x0 = x0 - lr * grad
    return x0