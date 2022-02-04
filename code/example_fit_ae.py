from LCO import LCO_TS, LCO_TS_stencil
import numpy as np
import copy
import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib as matplotlib
import pk_util as pk

# from pyoptsparse import OPT, Optimization

# color
my_blue = "#4C72B0"
my_red = "#C54E52"
my_green = "#56A968"
my_brown = "#b4943e"
my_purple = "#684c6b"
my_orange = "#cc5500"

# font
matplotlib.rcParams.update({"font.size": 20})
rc("font", **{"family": "sans-serif", "sans-serif": ["Helvetica"]})
rc("text", usetex=True)

isSubcritical = False


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


# -------------
# Common parameters
# -------------

ntimeinstance = 5
ndof = 4
angle_0 = 5  # Center point
pha_0 = 0.2
angle_arr = np.linspace(1, 13, 13)  # True curve sampling pts
if isSubcritical:
    x0 = [10.0, -1.5]
else:
    x0 = [10.0, 1.5]

# -------------
# Fitted curve
# -------------

delta_angle = 4.0
angle_list = [angle_0 - delta_angle, angle_0, angle_0 + delta_angle]
mag_list = copy.deepcopy(angle_list)
for i in range(len(mag_list)):
    mag_list[i] *= np.pi / 180.0

ind_p = 1

oscillator_list = []
mu_list = []
for i in range(3):

    # Extract the motion magnitude
    mag = mag_list[i]

    # Create the LCO object
    oscillator = LCO_TS(
        force, ntimeinstance, ndof, x=x0, pforcepx_func=pforcepx, pforcepw_func=pforcepw, pforcepmu_func=pforcepmu
    )

    # Set the prescribed motion
    oscillator.set_motion_mag_pha(mag, pha_0, ind_p=ind_p)

    # Generate an initial guess using Hopf bifurcation result
    Hopf_obj = pk.hopf_bifurcation(x0, get_A)
    mu_crit = Hopf_obj.solve(0.5, 1.4, 1e-4)
    Hopf_obj.compute_eigvec_magpha()
    xin = Hopf_obj.generate_init_sol(mag, pha_0, ntimeinstance, ind_p=ind_p)

    # Solve the LCO using the initial guess
    sol = oscillator.solve(xin)

    # Store the data
    mu_list.append(sol[0])

    # Store the object
    oscillator_list.append(oscillator)

LCO_stability = LCO_TS_stencil(oscillator_list, mag_list)
LCO_stability.compute_abd()

# Display the fitted curve
N_bif_fitted = 100
angle_bif_arr_fitted = np.linspace(1, 13, N_bif_fitted)
mag_bif_arr_fitted = angle_bif_arr_fitted * (np.pi / 180.0)
mu_bif_arr_fitted = np.zeros(N_bif_fitted)
for i in range(N_bif_fitted):
    mu_bif_arr_fitted[i] = LCO_stability.evaluate(mag_bif_arr_fitted[i])

# -------------
# True curve
# -------------

# Generate an initial guess using Hopf
mag_init = angle_arr[0] * (np.pi / 180.0)

Hopf_obj = pk.hopf_bifurcation(x0, get_A,)
mu_crit = Hopf_obj.solve(0.5, 1.4, 1e-4)
Hopf_obj.compute_eigvec_magpha()
sol = Hopf_obj.generate_init_sol(mag_init, pha_0, ntimeinstance, ind_p=ind_p)

# True curve

mu_arr = np.zeros_like(angle_arr)
j = 0
for angle in angle_arr:
    mag = angle * (np.pi / 180.0)
    oscillator = LCO_TS(
        force, ntimeinstance, ndof, x=x0, pforcepx_func=pforcepx, pforcepw_func=pforcepw, pforcepmu_func=pforcepmu
    )

    oscillator.set_motion_mag_pha(mag, pha_0, ind_p=ind_p)
    sol = oscillator.solve(sol)

    mu_arr[j] = sol[0]

    j += 1

# -------------
# Plot
# -------------

fig, ax = plt.subplots(1, figsize=(8, 4))

ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)

ax.set_xlabel(r"$\mu$", fontsize=20)
ax.set_ylabel(r"$\alpha, (\mathrm{deg})$", fontsize=20, rotation=0)

ax.yaxis.set_label_coords(-0.15, 0.4)

# ax.set_xlim(0.1, 1.1)
ax.set_ylim(0, 15.0)

# ax.text(0.18, 1.05, r"Fitted", fontsize=12, color=my_blue)
# ax.text(0.18, 0.90, r"Actual", fontsize=12, color=my_red, alpha = 0.5)
# ax.text(0.8, 0.55, r"Data", fontsize=12, color=my_red)

ax.plot(mu_list, angle_list, "o", color=my_red)
ax.plot(mu_bif_arr_fitted, angle_bif_arr_fitted, "-", color=my_blue)
ax.plot(mu_arr, angle_arr, "-x", color=my_red, alpha=0.5)

if isSubcritical:
    plt.savefig("../doc/figures/example_fit_ae_sub.pdf", bbox_inches="tight")
else:
    plt.savefig("../doc/figures/example_fit_ae_sup.pdf", bbox_inches="tight")

plt.show()
