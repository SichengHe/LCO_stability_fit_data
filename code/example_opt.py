from LCO import LCO_TS, LCO_TS_stencil
import numpy as np
import copy
import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib as matplotlib
import pk_util as pk
import example_simple_setting as dyn_setting
from plot_packages import *
from matplotlib import colors

ntimeinstance = 5
ndof = 2
mag_0 = 0.5
pha_0 = 0.2
x0 = [0.3, 1.0]
nx = len(x0)
delta_mag = 0.1
mag_list = [mag_0 - delta_mag, mag_0, mag_0 + delta_mag]

oscillator_list = []
for i in range(3):

    # Extract the motion magnitude
    mag = mag_list[i]

    # Create the LCO object
    oscillator = LCO_TS(
        dyn_setting.force,
        ntimeinstance,
        ndof,
        x=x0,
        pforcepx_func=dyn_setting.pforcepx,
        pforcepw_func=dyn_setting.pforcepw,
        pforcepmu_func=dyn_setting.pforcepmu,
    )

    # Set the prescribed motion
    oscillator.set_motion_mag_pha(mag, pha_0)

    # Generate an initial guess using Hopf bifurcation result
    Hopf_obj = pk.hopf_bifurcation(x0, dyn_setting.get_A, ndof=2)
    mu_crit = Hopf_obj.solve(0.5, 1.4, 1e-4)
    Hopf_obj.compute_eigvec_magpha()
    xin = Hopf_obj.generate_init_sol(mag, pha_0, ntimeinstance)

    # Solve the LCO using the initial guess
    sol = oscillator.solve(xin)

    # Store the object
    oscillator_list.append(oscillator)

# Fitting
LCO_stability = LCO_TS_stencil(oscillator_list, mag_list)
LCO_stability.compute_abd()

# ========================================
#    Optimization problem setup
# ========================================


def objfunc(xdict):

    global LCO_stability

    # Extract the design var
    x = xdict["xvars"]

    # Solve the equation
    LCO_stability.set_design_var(x)
    xin_list = LCO_stability.get_state_var()
    xin_list = LCO_stability.solve(xin_list)

    # Update parameter
    LCO_stability.reset_mu()
    LCO_stability.compute_abd()

    # Extract the objective function
    obj = -LCO_stability.get_mu()

    # Extract the constraint
    LCO_stability.compute_stability()
    con = LCO_stability.get_stability()

    # Set the objective function and con
    funcs = {}
    funcs["obj"] = obj
    funcs["con"] = con

    # Set failure flag
    fail = False

    return funcs, fail


def sens(xdict, funcs):

    global LCO_stability

    # Extract the design variable
    x = xdict["xvars"]
    nx = len(x)

    # Solve the equation
    LCO_stability.set_design_var(x)
    xin_list = LCO_stability.get_state_var()
    xin_list = LCO_stability.solve(xin_list)

    # Compute the objective derivative
    dmudx_adj = LCO_stability.compute_mu_der()

    # Compute the constraint derivative
    LCO_stability.compute_stability_der()
    dfdx = LCO_stability.get_stability_der()

    # Set the objective function and con derivative
    x = xdict["xvars"]
    funcsSens = {
        "obj": {"xvars": -dmudx_adj},
        "con": {"xvars": dfdx},
    }

    fail = False
    return funcsSens, fail


# ========================================
#    Optimization problem setup
# ========================================

from pyoptsparse import OPT, Optimization

# Optimization Object
optProb = Optimization("LCO parameter optimization with stability constraint", objfunc)

# Design Variables
lower = [0.0, 0.0]
upper = [1.0, 1.0]
value = x0
optProb.addVarGroup("xvars", 2, lower=lower, upper=upper, value=value)

# Constraints
lower = [0.1]
upper = [None]
optProb.addConGroup("con", 1, lower=lower, upper=upper)

# Objective
optProb.addObj("obj")

# Check optimization problem:
print(optProb)

# Optimizer
optimizer = "snopt"
optOptions = {}
opt = OPT(optimizer, options=optOptions)

# Solution
histFileName = "%s_LCO_Hist.hst" % optimizer

sol = opt(optProb, sens=sens, storeHistory=histFileName)

# Check Solution
print(sol)


# ========================================
#    Postprocess
# ========================================


def generate_LCO(x0, ntimeinstance, ndof, N, magmin, magmax, pha_0, force, pforcepx, pforcepw, pforcepmu):

    oscillator = LCO_TS(
        force, ntimeinstance, ndof, x=x0, pforcepx_func=pforcepx, pforcepw_func=pforcepw, pforcepmu_func=pforcepmu
    )
    oscillator.set_motion_mag_pha(magmin, pha_0)

    w0 = np.zeros(ntimeinstance * ndof)
    for i in range(ntimeinstance * ndof):
        w0[i] = mag_0 * np.sin(float(i) / float(ntimeinstance * ndof) * 2.0 * np.pi + pha_0)

    mu = 0.0
    T = 1.0
    omega = 2.0 * np.pi / T

    xin = np.zeros(ntimeinstance * ndof + 2)
    xin[0] = mu
    xin[1] = omega
    xin[2:] = w0[:]

    oscillator.solve(xin)

    magmin = magmin
    magmax = magmax

    mag_arr, mu_arr = oscillator.gen_bif_diag(xin, magmin, magmax, N)

    return mag_arr, mu_arr


x_init = np.array(x0)
x_final = np.array([1.0, 0.43])  # HACK!

N = 21
magmin = 0.1
magmax = 1.0
pha_0 = 0.2

NN = 17
mag_arr_list = []
mu_arr_list = []
for i in range(NN):

    print("i", i)

    weight = 1.0 - float(i) / (float(NN) - 1)
    x_init_inter = x_init * weight + x_final * (1 - weight)

    mag_arr, mu_arr = generate_LCO(
        x_init_inter,
        ntimeinstance,
        ndof,
        N,
        magmin,
        magmax,
        pha_0,
        dyn_setting.force,
        dyn_setting.pforcepx,
        dyn_setting.pforcepw,
        dyn_setting.pforcepmu,
    )

    mag_arr_list.append(mag_arr)
    mu_arr_list.append(mu_arr)


fig, axs = plt.subplots(1, figsize=(8, 4))


def plot(ax, mu_arr, mag_arr, color):

    # Bifurcation diagram
    axs.plot(mu_arr, mag_arr, "-", color=color)

    return ax


rgb_blue = colors.to_rgba(my_blue)
rgb_green = colors.to_rgba(my_green)
for i in range(NN):

    weight = 1.0 - float(i) / float(NN)

    my_color_arr = np.zeros(3)
    for j in range(3):
        my_color_arr[j] = rgb_blue[j] * weight + rgb_green[j] * (1 - weight)

    my_color = (my_color_arr[0], my_color_arr[1], my_color_arr[2], 1.0)

    mag_arr = mag_arr_list[i]
    mu_arr = mu_arr_list[i]

    axs = plot(axs, mu_arr, mag_arr, my_color)

axs.spines["right"].set_visible(False)
axs.spines["top"].set_visible(False)

axs.set_xlabel(r"$\mu$", fontsize=20)
axs.set_ylabel(r"$\hat{\alpha}$", fontsize=20, rotation=0)
axs.yaxis.set_label_coords(-0.175, 0.4)

mu_arr_init = mu_arr_list[0]
mu_arr_final = mu_arr_list[-1]
mag_arr_init = mag_arr_list[0]
mag_arr_final = mag_arr_list[-1]


min_mu = min(min(mu_arr_init), min(mu_arr_final))
max_mu = max(max(mu_arr_init), max(mu_arr_final))
delta_mu = max_mu - min_mu
axs.set_xlim([min_mu - delta_mu * 0.1, max_mu + delta_mu * 0.1])
axs.set_ylim([0, magmax * 1.1])

# Additional lines
index = 9
axs.plot([min_mu - delta_mu * 0.1, mu_arr_final[index]], [0.5, 0.5], color="k", alpha=0.2)
axs.plot([mu_arr_init[index], mu_arr_init[index]], [mag_arr_init[index], 0], color=my_blue, alpha=0.2)
axs.plot([mu_arr_final[index], mu_arr_final[index]], [mag_arr_final[index], 0], color=my_green, alpha=0.2)

axs.text(0.48, 0.6, r"Baseline", fontsize=12, color=my_blue)
axs.text(0.8, 0.6, r"Optimized", fontsize=12, color=my_green)

# plt.show()
plt.savefig("../doc/figures/example_opt.pdf", bbox_inches="tight")
