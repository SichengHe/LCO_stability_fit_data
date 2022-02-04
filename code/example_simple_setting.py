import numpy as np

def force(mu, w, x):

    """
        Forcing term.
    """

    f0 = (mu - x[0]) * w[0] - w[1] + (2.0 * x[0] * x[1] - 1.0) * w[0] ** 3
    f1 = w[0] + (mu - x[1]) * w[1] + (2.0 * x[1] - 1.0) * w[1] ** 3

    f = np.zeros(2)
    f[0] = f0
    f[1] = f1

    return f


def pforcepw(mu, w, x):

    """
        pf / pw
    """

    # Each entry ...
    pf0pw0 = (mu - x[0]) + (2.0 * x[0] * x[1] - 1.0) * 3 * w[0] ** 2
    pf0pw1 = -1.0
    pf1pw0 = 1.0
    pf1pw1 = (mu - x[1]) + (2.0 * x[1] - 1.0) * 3 * w[1] ** 2

    # Fill in
    pfpw = np.zeros((2, 2))

    pfpw[0, 0] = pf0pw0
    pfpw[0, 1] = pf0pw1
    pfpw[1, 0] = pf1pw0
    pfpw[1, 1] = pf1pw1

    return pfpw


def pforcepx(mu, w, x):

    """
        pf / px
    """

    # Each entry ...
    pf0px0 = -w[0] + 2.0 * x[1] * w[0] ** 3
    pf0px1 = 2.0 * x[0] * w[0] ** 3
    pf1px0 = 0.0
    pf1px1 = -w[1] + 2.0 * w[1] ** 3

    # Fill in
    pfpx = np.zeros((2, 2))

    pfpx[0, 0] = pf0px0
    pfpx[0, 1] = pf0px1
    pfpx[1, 0] = pf1px0
    pfpx[1, 1] = pf1px1

    return pfpx


def pforcepmu(mu, w, x):

    """
        pf / pmu
    """

    # Each entry ...
    pf0pmu = w[0]
    pf1pmu = w[1]

    # Fill in
    pfpmu = np.zeros(2)

    pfpmu[0] = pf0pmu
    pfpmu[1] = pf1pmu

    return pfpmu


def get_A(mu, x):

    """
        Get the linear Jacobian matrix.
    """

    A = np.zeros((2, 2))
    A[0, 0] = mu - 1.0
    A[0, 1] = -1.0
    A[1, 0] = 1.0
    A[1, 1] = mu - 1.0

    return A
