from LCO import LCO_TS, LCO_TS_stencil
import numpy as np
import copy
import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib as matplotlib
import pk_util as pk

# color
my_blue = '#4C72B0'
my_red = '#C54E52'
my_green = '#56A968'
my_brown = '#b4943e'
my_purple = '#684c6b'
my_orange = '#cc5500'

# font
matplotlib.rcParams.update({'font.size': 20})
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)

def force(mu, w, x):

    '''
        Forcing term.
    '''

    f0 = (mu - x[0]) * w[0] - w[1] + (2.0 * x[0] - 1.0) * w[0] ** 3
    f1 = w[0] + (mu - x[1]) * w[1] + (2.0 * x[1] - 1.0) * w[1] ** 3

    f = np.zeros(2)
    f[0] = f0
    f[1] = f1

    return f

def pforcepw(mu, w, x):

    '''
        pf / pw
    '''

    # Each entry ...
    pf0pw0 = (mu - x[0]) + (2.0 * x[0] - 1.0) * 3 * w[0] ** 2
    pf0pw1 = - 1.0
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

    '''
        pf / px
    '''

    # Each entry ...
    pf0px0 = - w[0] + 2.0 * w[0] ** 3
    pf0px1 = 0.0
    pf1px0 = 0.0
    pf1px1 = - w[1] + 2.0 * w[1] ** 3

    # Fill in
    pfpx = np.zeros((2, 2))

    pfpx[0, 0] = pf0px0
    pfpx[0, 1] = pf0px1
    pfpx[1, 0] = pf1px0
    pfpx[1, 1] = pf1px1

    return pfpx

def pforcepmu(mu, w, x):

    '''
        pf / pmu
    '''

    # Each entry ...
    pf0pmu = w[0]
    pf1pmu = w[1]

    # Fill in
    pfpmu = np.zeros(2)

    pfpmu[0] = pf0pmu
    pfpmu[1] = pf1pmu

    return pfpmu

def get_A(mu, x):

    '''
        Get the linear Jacobian matrix.
    '''

    A = np.zeros((2, 2))
    A[0, 0] = mu - 1.0
    A[0, 1] = - 1.0
    A[1, 0] = 1.0
    A[1, 1] = mu - 1.0

    return A

ntimeinstance = 5
ndof = 2
mag_0 = 0.5
pha_0 = 0.2
x0 = [1.0, 1.0]
nx = len(x0)
delta_mag = 0.1
mag_list = [mag_0 - delta_mag, \
mag_0, \
mag_0 + delta_mag]


oscillator_list = []
mu_list = []
for i in range(3):

    # Extract the motion magnitude
    mag = mag_list[i]

    # Create the LCO object
    oscillator = LCO_TS(force, ntimeinstance, ndof, x = x0, \
        pforcepx_func = pforcepx, pforcepw_func = pforcepw, \
        pforcepmu_func = pforcepmu)

    # Set the prescribed motion
    oscillator.set_motion_mag_pha(mag, pha_0)

    # Generate an initial guess using Hopf bifurcation result
    Hopf_obj = pk.hopf_bifurcation(x0, get_A, ndof = 2)
    mu_crit = Hopf_obj.solve(0.5, 1.4, 1e-4)
    Hopf_obj.compute_eigvec_magpha()
    xin = Hopf_obj.generate_init_sol(mag, pha_0, ntimeinstance)

    # Solve the LCO using the initial guess
    sol = oscillator.solve(xin)

    # Store the data
    mu_list.append(sol[0])

    # Store the object
    oscillator_list.append(oscillator)

# Fitting
LCO_stability = LCO_TS_stencil(oscillator_list, mag_list)
LCO_stability.compute_abd()

# Display the fitted curve
N_bif_fitted = 100
mag_bif_arr_fitted = np.linspace(0, 1, N_bif_fitted)
mu_bif_arr_fitted = np.zeros(N_bif_fitted)
for i in range(N_bif_fitted):
    mu_bif_arr_fitted[i] = LCO_stability.evaluate(mag_bif_arr_fitted[i])

# True curve
mag_min = 0.1
mag_max = 1.0
N_bif_true = 19
oscillator = LCO_TS(force, ntimeinstance, ndof, x = x0, \
        pforcepx_func = pforcepx, pforcepw_func = pforcepw, \
        pforcepmu_func = pforcepmu)
oscillator.set_motion_mag_pha(mag_min, pha_0)
xin = oscillator.generate_xin()
sol = oscillator.solve(xin)

mag_bif_arr_true, mu_bif_arr_true = oscillator.gen_bif_diag(sol, mag_min, mag_max, N_bif_true)


fig, ax = plt.subplots(1, figsize=(8,4))

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

ax.set_xlabel(r'$\mu$', fontsize=20)
ax.set_ylabel(r'$\hat{\alpha}$', fontsize=20, rotation=0)

ax.yaxis.set_label_coords(-0.15,0.4)

ax.set_xlim(0.1, 1.1)
ax.set_ylim(0, 1.1)

ax.text(0.18, 1.05, r"Fitted", fontsize=12, color=my_blue)
ax.text(0.18, 0.90, r"Actual", fontsize=12, color=my_red, alpha = 0.5)
ax.text(0.8, 0.55, r"Data", fontsize=12, color=my_red)

ax.plot(mu_bif_arr_fitted, mag_bif_arr_fitted, '-', color=my_blue)
ax.plot(mu_bif_arr_true, mag_bif_arr_true, '-x', color=my_red, alpha = 0.5)
ax.plot(mu_list, mag_list, 'o', color=my_red)

plt.savefig('../doc/figures/example_fit.pdf',bbox_inches='tight')

plt.show()



