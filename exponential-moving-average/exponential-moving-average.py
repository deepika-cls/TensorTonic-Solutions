def exponential_moving_average(values, alpha):
    """
    Compute the exponential moving average of the given values.
    """
    res = [values[0]]
    prev_data = values[0]
    for i in range(1, len(values)):
        prev_data = values[i] * alpha + prev_data * (1-alpha)
        res.append(prev_data)
    return res