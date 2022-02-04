from LCO import LCO_TS, LCO_TS_stencil
import numpy as np
import copy
import pk_util as pk

def force(mu, w, x):

    '''
        Forcing term.
    '''

    # f0 = mu * w[0] - w[1] + (- 1.0 * (1 - x[0]) + 1.0 * x[0]) * w[0] ** 3
    # f1 = w[0] + mu * w[1] + (- 1.0 * (1 - x[1]) + 1.0 * x[1]) * w[1] ** 3
    f0 = (mu - x[0]) * w[0] - w[1] + (2.0 * x[0] * x[1] - 1.0) * w[0] ** 3
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
    # pf0pw0 = mu + (- 1.0 * (1 - x[0]) + 1.0 * x[0]) * 3 * w[0] ** 2
    # pf0pw1 = - 1.0
    # pf1pw0 = 1.0
    # pf1pw1 = mu + (- 1.0 * (1 - x[1]) + 1.0 * x[1]) * 3 * w[1] ** 2
    pf0pw0 = (mu - x[0]) + (2.0 * x[0] * x[1] - 1.0) * 3.0 * w[0] ** 2
    pf0pw1 = - 1.0
    pf1pw0 = 1.0
    pf1pw1 = (mu - x[1]) + (2.0 * x[1] - 1.0) * 3.0 * w[1] ** 2

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

    # f0 = mu * w[0] - w[1] + (- 1.0 * (1 - x) + 1.0 * x) * w[0] ** 3
    # f1 = w[0] + mu * w[1]

    # Each entry ...
    # pf0px0 = (- 1.0 * (- 1.0) + 1.0) * w[0] ** 3
    # pf0px1 = 0.0
    # pf1px0 = 0.0
    # pf1px1 = (- 1.0 * (- 1.0) + 1.0) * w[1] ** 3
    pf0px0 = - w[0] + 2.0 * x[1] * w[0] ** 3
    pf0px1 = 2.0 * x[0] * w[0] ** 3
    pf1px0 = 0.0
    pf1px1 = - w[1] + 2.0 * w[1] ** 3

    # Fill in
    pfpx = np.zeros((2, 2))

    pfpx[0, 0] = pf0px0
    pfpx[1, 0] = pf1px0
    pfpx[0, 1] = pf0px1
    pfpx[1, 1] = pf1px1

    return pfpx

def pforcepmu(mu, w, x):

    '''
        pf / pmu
    '''

    # f0 = mu * w[0] - w[1] + (- 1.0 * (1 - x) + 1.0 * x) * w[0] ** 3
    # f1 = w[0] + mu * w[1]

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
    A[0, 0] = mu - x[0]
    A[0, 1] = - 1.0
    A[1, 0] = 1.0
    A[1, 1] = mu - x[1]

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
for i in range(3):
    mag = mag_list[i]
    oscillator = LCO_TS(force, ntimeinstance, ndof, x = x0, \
        pforcepx_func = pforcepx, pforcepw_func = pforcepw, \
        pforcepmu_func = pforcepmu)
    oscillator.set_motion_mag_pha(mag, pha_0)

    Hopf_obj = pk.hopf_bifurcation(x0, get_A, ndof = 2)
    mu_crit = Hopf_obj.solve(0.5, 1.4, 1e-4)
    Hopf_obj.compute_eigvec_magpha()
    xin = Hopf_obj.generate_init_sol(mag, pha_0, ntimeinstance)

    sol = oscillator.solve(xin)

    oscillator_list.append(oscillator)

# Ajoint fitted
LCO_stability = LCO_TS_stencil(oscillator_list, mag_list)
LCO_stability.compute_stability()
stability_measure_0 = LCO_stability.get_stability()
LCO_stability.compute_stability_der()
stability_measure_der_adjoint = LCO_stability.get_stability_der()

# FD fitted
stability_measure_der_FD = np.zeros(nx)
epsilon = 1e-6
for i in range(nx):
    x0_per = copy.deepcopy(x0)
    x0_per[i] += epsilon

    LCO_stability.set_design_var(x0_per)
    xin_list = LCO_stability.get_state_var()
    xin_list = LCO_stability.solve(xin_list)

    # Update parameter
    LCO_stability.reset_mu()
    LCO_stability.compute_abd()

    LCO_stability.compute_stability()
    stability_measure_per = LCO_stability.get_stability()

    stability_measure_der_FD[i] = (stability_measure_per - stability_measure_0) / epsilon

# FD actual
epsilon_actual = 1e-6
mag_actual_list = [mag_0 - epsilon_actual, mag_0, mag_0 + epsilon_actual]
mu_actual_list = []
for i in range(3):
    mag = mag_actual_list[i]
    oscillator = LCO_TS(force, ntimeinstance, ndof, x = x0, \
        pforcepx_func = pforcepx, pforcepw_func = pforcepw, \
        pforcepmu_func = pforcepmu)
    oscillator.set_motion_mag_pha(mag, pha_0)

    Hopf_obj = pk.hopf_bifurcation(x0, get_A, ndof = 2)
    mu_crit = Hopf_obj.solve(0.5, 1.4, 1e-4)
    Hopf_obj.compute_eigvec_magpha()
    xin = Hopf_obj.generate_init_sol(mag, pha_0, ntimeinstance)

    sol = oscillator.solve(xin)

    mu_actual_list.append(sol[0])


stability_measure_0_FD = (mu_actual_list[-1] - mu_actual_list[0]) / (2 * epsilon_actual)

print("="*20)
print("-"*20)
print("Curve slope")
print("-"*20)
print("Fitted", stability_measure_0)
print("FD", stability_measure_0_FD)
print("-"*20)

print("-"*20)
print("Total derivative")
print("-"*20)
print("Adjoint", stability_measure_der_adjoint)
print("FD", stability_measure_der_FD)
print("="*20)



if 1:
    # Test LCO dmu / dx.

    dmudx_adj = oscillator.solve_bot_dmudx()


    print("dmudx_adj", dmudx_adj)

    epsilon = 1e-7
    x0_per = copy.deepcopy(x0)
    x0_per[0] += epsilon
    oscillator.set_design_var(x0_per)
    sol1 = oscillator.solve(xin)
    print("sol1", sol1)

    dmudx_FD = (sol1[0] - sol[0]) / epsilon
    print("dmudx_FD", dmudx_FD)
