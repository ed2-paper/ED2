"""Dummy MPI utilities.

Note that we currently don't use any distributed computing features. However, we
might do in the future. To stay compatible use these mock MPI functions.
"""

import numpy as np


def proc_id():
    """Get rank of calling process."""
    return 0


def mpi_statistics_scalar(data, with_min_and_max=False):
    """Get mean/std and optional min/max of scalar data across MPI processes.

    Args:
        data: An array containing samples of the scalar to produce statistics
            for.
        with_min_and_max (bool): If true, return min and max of x in
            addition to mean and std.

    Returns:
        Aggregate statistics.
    """
    mean = np.mean(data)
    std = np.std(data)
    if with_min_and_max:
        minimum = np.min(data)
        maximum = np.max(data)
        return mean, std, minimum, maximum
    return mean, std


def mpi_histogram(data):
    """Get mean/std and optional min/max of scalar data across MPI processes.

    Args:
        data (list): An array containing samples to produce histogram for.

    Returns:
        numpy.ndarray: The values of the histogram.
        numpy.ndarray: The bin edges (length(hist)+1).
    """
    return np.histogram(data, bins=20, density=True)
