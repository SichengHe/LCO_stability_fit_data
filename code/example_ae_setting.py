import numpy as np

isSubcritical = True

def force(mu, w, x):

    """
        Forcing term.
    """

    global isSubcritical

    # Define constants
    ebar_con = 0.2
    if isSubcritical:
        kappa_5_con = 50.0
    else:
        kappa_5_con = 0.0
    Omega_con = 0.5
    ra_con = 0.3
    xa_con = 0.2

    # Extract design var
    mbar = x[0]
    kappa_3 = x[1]

    # Extract state var
    hbar = w[0]
    alpha = w[1]
    hbar_dot = w[2]
    alpha_dot = w[3]

    # Linear force
    A = np.zeros((4, 4))
    A[0, 2] = 1.0
    A[1, 3] = 1.0

    # Ms^{-1} * Ks
    MsInv_Ks = np.zeros((2, 2))
    coeff = 1.0 / (ra_con ** 2 - xa_con ** 2)
    MsInv_Ks[0, 0] = coeff * ra_con ** 2 * Omega_con ** 2
    MsInv_Ks[0, 1] = coeff * (-xa_con * ra_con ** 2)
    MsInv_Ks[1, 0] = coeff * (-xa_con * Omega_con ** 2)
    MsInv_Ks[1, 1] = coeff * ra_con ** 2

    # Ms^{-1} * Ka
    MsInv_Ka = np.zeros((2, 2))
    coeff = 2.0 * mu ** 2 / ((ra_con ** 2 - xa_con ** 2) * mbar)
    MsInv_Ka[0, 1] = coeff * (ra_con ** 2 + xa_con * ebar_con)
    MsInv_Ka[1, 1] = coeff * (-xa_con - ebar_con)

    # Ms^{-1} * Da
    MsInv_Da = np.zeros((2, 2))
    coeff = 2.0 * mu / ((ra_con ** 2 - xa_con ** 2) * mbar)
    MsInv_Da[0, 0] = coeff * (ra_con ** 2 + xa_con * ebar_con)
    MsInv_Da[1, 0] = coeff * (-xa_con - ebar_con)

    A[2:4, 0:2] += -1.0 * MsInv_Ks[:, :]
    A[2:4, 0:2] += -1.0 * MsInv_Ka[:, :]
    A[2:4, 2:4] += -1.0 * MsInv_Da[:, :]

    # Nonlinear force
    Fnl = np.zeros(4)
    coeff = (ra_con ** 2 * (kappa_3 * alpha ** 3 + kappa_5_con * alpha ** 5)) / (ra_con ** 2 - xa_con ** 2)
    Fnl[2] = coeff * xa_con
    Fnl[3] = coeff * (-1.0)

    # Total force
    f = np.zeros(4)
    f[:] += A.dot(w) + Fnl[:]

    return f


def pforcepw(mu, w, x):

    """
        pf / pw
    """

    # Define constants
    ebar_con = 0.2

    if isSubcritical:
        kappa_5_con = 50.0
    else:
        kappa_5_con = 0.0
    Omega_con = 0.5
    ra_con = 0.3
    xa_con = 0.2

    # Extract design var
    mbar = x[0]
    kappa_3 = x[1]

    # Extract state var
    hbar = w[0]
    alpha = w[1]
    hbar_dot = w[2]
    alpha_dot = w[3]

    # Linear force
    A = np.zeros((4, 4))
    A[0, 2] = 1.0
    A[1, 3] = 1.0

    # Ms^{-1} * Ks
    MsInv_Ks = np.zeros((2, 2))
    coeff = 1.0 / (ra_con ** 2 - xa_con ** 2)
    MsInv_Ks[0, 0] = coeff * ra_con ** 2 * Omega_con ** 2
    MsInv_Ks[0, 1] = coeff * (-xa_con * ra_con ** 2)
    MsInv_Ks[1, 0] = coeff * (-xa_con * Omega_con ** 2)
    MsInv_Ks[1, 1] = coeff * ra_con ** 2

    # Ms^{-1} * Ka
    MsInv_Ka = np.zeros((2, 2))
    coeff = 2.0 * mu ** 2 / ((ra_con ** 2 - xa_con ** 2) * mbar)
    MsInv_Ka[0, 1] = coeff * (ra_con ** 2 + xa_con * ebar_con)
    MsInv_Ka[1, 1] = coeff * (-xa_con - ebar_con)

    # Ms^{-1} * Da
    MsInv_Da = np.zeros((2, 2))
    coeff = 2.0 * mu / ((ra_con ** 2 - xa_con ** 2) * mbar)
    MsInv_Da[0, 0] = coeff * (ra_con ** 2 + xa_con * ebar_con)
    MsInv_Da[1, 0] = coeff * (-xa_con - ebar_con)

    A[2:4, 0:2] += -1.0 * MsInv_Ks[:, :]
    A[2:4, 0:2] += -1.0 * MsInv_Ka[:, :]
    A[2:4, 2:4] += -1.0 * MsInv_Da[:, :]

    # Nonlinear force
    pFnlpw = np.zeros((4, 4))
    coeff = (ra_con ** 2 * (kappa_3 * 3.0 * alpha ** 2 + kappa_5_con * 5.0 * alpha ** 4)) / (ra_con ** 2 - xa_con ** 2)
    pFnlpw[2, 1] = coeff * xa_con
    pFnlpw[3, 1] = coeff * (-1.0)

    # Total force
    pfpw = np.zeros((4, 4))
    pfpw[:, :] += A[:, :]
    pfpw[:, :] += pFnlpw[:, :]

    return pfpw


def pforcepx(mu, w, x):

    """
        p f / p x
    """

    global isSubcritical

    # Define constants
    ebar_con = 0.2
    if isSubcritical:
        kappa_5_con = 50.0
    else:
        kappa_5_con = 0.0
    Omega_con = 0.5
    ra_con = 0.3
    xa_con = 0.2

    # Extract design var
    mbar = x[0]
    kappa_3 = x[1]

    # Extract state var
    hbar = w[0]
    alpha = w[1]
    hbar_dot = w[2]
    alpha_dot = w[3]

    # Linear force
    pA_pmbar = np.zeros((4, 4))

    # Ms^{-1} * Ka
    pMsInv_Ka_pmbar = np.zeros((2, 2))
    coeff = -2.0 * mu ** 2 / ((ra_con ** 2 - xa_con ** 2) * mbar ** 2)
    pMsInv_Ka_pmbar[0, 1] = coeff * (ra_con ** 2 + xa_con * ebar_con)
    pMsInv_Ka_pmbar[1, 1] = coeff * (-xa_con - ebar_con)

    # Ms^{-1} * Da
    pMsInv_Da_pmbar = np.zeros((2, 2))
    coeff = -2.0 * mu / ((ra_con ** 2 - xa_con ** 2) * mbar ** 2)
    pMsInv_Da_pmbar[0, 0] = coeff * (ra_con ** 2 + xa_con * ebar_con)
    pMsInv_Da_pmbar[1, 0] = coeff * (-xa_con - ebar_con)

    pA_pmbar[2:4, 0:2] += -1.0 * pMsInv_Ka_pmbar[:, :]
    pA_pmbar[2:4, 2:4] += -1.0 * pMsInv_Da_pmbar[:, :]

    # Nonlinear force
    pFnl_pkappa_3 = np.zeros(4)
    coeff = (ra_con ** 2 * (alpha ** 3)) / (ra_con ** 2 - xa_con ** 2)
    pFnl_pkappa_3[2] = coeff * xa_con
    pFnl_pkappa_3[3] = coeff * (-1.0)

    # Total force
    pfpx = np.zeros((4, 2))
    # pf / pmbar
    pfpx[:, 0] += pA_pmbar.dot(w)[:]
    # pf / p kappa_3
    pfpx[:, 1] += pFnl_pkappa_3[:]

    return pfpx


def pforcepmu(mu, w, x):

    """
        p f / p mu
    """

    global isSubcritical

    # Define constants
    ebar_con = 0.2
    if isSubcritical:
        kappa_5_con = 50.0
    else:
        kappa_5_con = 0.0
    Omega_con = 0.5
    ra_con = 0.3
    xa_con = 0.2

    # Extract design var
    mbar = x[0]
    kappa_3 = x[1]

    # Extract state var
    hbar = w[0]
    alpha = w[1]
    hbar_dot = w[2]
    alpha_dot = w[3]

    # Linear force
    pA_pmu = np.zeros((4, 4))

    # Ms^{-1} * Ka
    pMsinv_Ka_pmu = np.zeros((2, 2))
    coeff = 4.0 * mu / ((ra_con ** 2 - xa_con ** 2) * mbar)
    pMsinv_Ka_pmu[0, 1] = coeff * (ra_con ** 2 + xa_con * ebar_con)
    pMsinv_Ka_pmu[1, 1] = coeff * (-xa_con - ebar_con)

    # Ms^{-1} * Da
    pMsinv_Da_pmu = np.zeros((2, 2))
    coeff = 2.0 / ((ra_con ** 2 - xa_con ** 2) * mbar)
    pMsinv_Da_pmu[0, 0] = coeff * (ra_con ** 2 + xa_con * ebar_con)
    pMsinv_Da_pmu[1, 0] = coeff * (-xa_con - ebar_con)

    pA_pmu[2:4, 0:2] += -1.0 * pMsinv_Ka_pmu[:, :]
    pA_pmu[2:4, 2:4] += -1.0 * pMsinv_Da_pmu[:, :]

    # Total force
    pfpmu = np.zeros(4)
    pfpmu[:] += pA_pmu.dot(w)[:]

    return pfpmu


def get_A(mu, x):

    """
        Forcing term.
    """

    global isSubcritical

    # Define constants
    ebar_con = 0.2

    if isSubcritical:
        kappa_5_con = 50.0
    else:
        kappa_5_con = 0.0
    Omega_con = 0.5
    ra_con = 0.3
    xa_con = 0.2

    # Extract design var
    mbar = x[0]
    kappa_3 = x[1]

    # Linear force
    A = np.zeros((4, 4))
    A[0, 2] = 1.0
    A[1, 3] = 1.0

    # Ms^{-1} * Ks
    MsInv_Ks = np.zeros((2, 2))
    coeff = 1.0 / (ra_con ** 2 - xa_con ** 2)
    MsInv_Ks[0, 0] = coeff * ra_con ** 2 * Omega_con ** 2
    MsInv_Ks[0, 1] = coeff * (-xa_con * ra_con ** 2)
    MsInv_Ks[1, 0] = coeff * (-xa_con * Omega_con ** 2)
    MsInv_Ks[1, 1] = coeff * ra_con ** 2

    # Ms^{-1} * Ka
    MsInv_Ka = np.zeros((2, 2))
    coeff = 2.0 * mu ** 2 / ((ra_con ** 2 - xa_con ** 2) * mbar)
    MsInv_Ka[0, 1] = coeff * (ra_con ** 2 + xa_con * ebar_con)
    MsInv_Ka[1, 1] = coeff * (-xa_con - ebar_con)

    # Ms^{-1} * Da
    MsInv_Da = np.zeros((2, 2))
    coeff = 2.0 * mu / ((ra_con ** 2 - xa_con ** 2) * mbar)
    MsInv_Da[0, 0] = coeff * (ra_con ** 2 + xa_con * ebar_con)
    MsInv_Da[1, 0] = coeff * (-xa_con - ebar_con)

    A[2:4, 0:2] += -1.0 * MsInv_Ks[:, :]
    A[2:4, 0:2] += -1.0 * MsInv_Ka[:, :]
    A[2:4, 2:4] += -1.0 * MsInv_Da[:, :]

    return A
