import numpy as np


def moving_average(a, n):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def is_monotonic(arr, tolerance=1e-6):
    differences = [arr[i + 1] - arr[i] for i in range(len(arr) - 1)]
    is_uniform_difference = all(abs(diff - differences[0]) < tolerance for diff in differences)
    return is_uniform_difference
