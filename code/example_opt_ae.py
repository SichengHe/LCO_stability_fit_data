from LCO import LCO_TS, LCO_TS_stencil
import numpy as np
import copy
import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib as matplotlib
import pk_util as pk
import example_ae_setting as dyn_setting
from plot_packages import *
from matplotlib import colors

# ========================================
#    Problem setup
# ========================================

ntimeinstance = 5
ndof = 4
angle_0 = 5.0
mag_0 = angle_0 / 180.0 * np.pi
pha_0 = 0.2
x0 = [15.0, -3.0]
nx = len(x0)
delta_angle = 4.0
angle_list = [angle_0 - delta_angle, angle_0, angle_0 + delta_angle]
mag_list = copy.deepcopy(angle_list)
for i in range(3):
    mag_list[i] *= np.pi / 180.0
ind_p = 1

oscillator_list = []
sol_list = []
for i in range(3):
    mag = mag_list[i]
    oscillator = LCO_TS(
        dyn_setting.force,
        ntimeinstance,
        ndof,
        x=x0,
        pforcepx_func=dyn_setting.pforcepx,
        pforcepw_func=dyn_setting.pforcepw,
        pforcepmu_func=dyn_setting.pforcepmu,
    )
    oscillator.set_motion_mag_pha(mag, pha_0, ind_p=ind_p)

    # Generate an initial guess using Hopf bifurcation result
    Hopf_obj = pk.hopf_bifurcation(x0, dyn_setting.get_A)
    mu_crit = Hopf_obj.solve(0.5, 1.4, 1e-4)
    Hopf_obj.compute_eigvec_magpha()
    xin = Hopf_obj.generate_init_sol(mag, pha_0, ntimeinstance, ind_p=ind_p)

    sol = oscillator.solve(xin)

    oscillator_list.append(oscillator)
    sol_list.append(sol)

# Ajoint fitted
LCO_stability = LCO_TS_stencil(oscillator_list, mag_list)
LCO_stability.compute_stability()
stab = LCO_stability.get_stability()

# ========================================
#    Optimization problem setup
# ========================================


def objfunc(xdict):

    global LCO_stability

    # Extract the design var
    x = xdict["xvars"]

    mass = x[0] - x[1] ** 2

    # Solve the equation
    LCO_stability.set_design_var(x)
    xin_list = LCO_stability.get_state_var()
    xin_list = LCO_stability.solve(xin_list)

    # Update parameter
    LCO_stability.reset_mu()
    LCO_stability.compute_abd()

    # Extract the objective function
    mu = LCO_stability.get_mu()

    # Extract the constraint
    LCO_stability.compute_stability()
    stability = LCO_stability.get_stability()

    # Set the objective function and con
    funcs = {}

    funcs["obj_mass"] = mass
    funcs["con_mu"] = mu
    funcs["con_stablity"] = stability

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

    # Update parameter
    LCO_stability.reset_mu()
    LCO_stability.compute_abd()

    # Compute the objective derivative
    dmudx_adj = LCO_stability.compute_mu_der()

    # Compute the constraint derivative
    LCO_stability.compute_stability_der()
    dfdx = LCO_stability.get_stability_der()

    # Set the objective function and con derivative
    x = xdict["xvars"]
    funcsSens = {
        "obj_mass": {"xvars": [1.0, -x[1]]},
        "con_mu": {"xvars": dmudx_adj},
        "con_stablity": {"xvars": dfdx},
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
lower = [5.0, -3.0]
upper = [15.0, 0.0]
value = x0
optProb.addVarGroup("xvars", 2, lower=lower, upper=upper, value=value)

# Constraints
optProb.addCon("con_mu", lower=0.8)
optProb.addCon("con_stablity", lower=0.02)

# Objective
optProb.addObj("obj_mass")

# Check optimization problem:
print(optProb)

# Optimizer
optimizer = "snopt"
optOptions = {}
opt = OPT(optimizer, options=optOptions)

# Solution
histFileName = "%s_LCO_ae_Hist.hst" % optimizer

sol = opt(optProb, sens=sens, storeHistory=histFileName)

# Check Solution
print(sol)

# ========================================
#    Postprocess
# ========================================


def generate_LCO(x0, ntimeinstance, ndof, N, magmin, magmax, pha_0, force, pforcepx, pforcepw, pforcepmu, get_A):

    oscillator = LCO_TS(
        force, ntimeinstance, ndof, x=x0, pforcepx_func=pforcepx, pforcepw_func=pforcepw, pforcepmu_func=pforcepmu
    )
    oscillator.set_motion_mag_pha(magmin, pha_0, ind_p=ind_p)

    Hopf_obj = pk.hopf_bifurcation(x0, get_A)
    mu_crit = Hopf_obj.solve(0.5, 1.4, 1e-4)
    Hopf_obj.compute_eigvec_magpha()
    xin = Hopf_obj.generate_init_sol(magmin, pha_0, ntimeinstance, ind_p=ind_p)

    oscillator.solve(xin)

    magmin = magmin
    magmax = magmax

    mag_arr, mu_arr = oscillator.gen_bif_diag(xin, magmin, magmax, N)

    return mag_arr, mu_arr


x_init = np.array(x0)
x_final = np.array([9.24741, -0.37354])  # HACK!

N = 27
angle_array = np.linspace(0.1, 13.0, N)
magmin = angle_array[0] / 180.0 * np.pi
magmax = angle_array[-1] / 180.0 * np.pi
pha_0 = 0.2

NN = 25
mag_arr_list = []
mu_arr_list = []
for i in range(NN):

    print("=" * 20)
    print("Running the ", i, "-th design variable.")
    print("=" * 20)

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
        dyn_setting.get_A,
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

    mag_arr = mag_arr_list[i] * 180.0 / np.pi
    mu_arr = mu_arr_list[i]

    axs = plot(axs, mu_arr, mag_arr, my_color)

axs.spines["right"].set_visible(False)
axs.spines["top"].set_visible(False)


axs.set_xlabel(r"$\mu$", fontsize=20)
axs.set_ylabel(r"$\alpha, \,\, (\mathrm{deg})$", fontsize=20, rotation=0)
axs.yaxis.set_label_coords(-0.175, 0.4)

mu_arr_init = mu_arr_list[0]
mu_arr_final = mu_arr_list[-1]
mag_arr_init = mag_arr_list[0]
mag_arr_final = mag_arr_list[-1]

min_mu = min(min(mu_arr_init), min(mu_arr_final))
max_mu = max(max(mu_arr_init), max(mu_arr_final))
delta_mu = max_mu - min_mu
axs.set_xlim([min_mu - delta_mu * 0.2, max_mu + delta_mu * 0.1])
axs.set_ylim([0, angle_array[-1] * 1.1])

# Additional lines
index = 10
axs.plot([min_mu - delta_mu * 0.2, mu_arr_init[index]], [angle_array[index], angle_array[index]], color="k", alpha=0.2)
axs.plot([mu_arr_init[index], mu_arr_init[index]], [angle_array[index], 0], color=my_blue, alpha=0.2)
axs.plot([mu_arr_final[index], mu_arr_final[index]], [angle_array[index], 0], color=my_green, alpha=0.2)

axs.text(0.77, 11.0, r"Optimized", fontsize=12, color=my_green)
axs.text(1.02, 11.0, r"Baseline", fontsize=12, color=my_blue)

# plt.show()
plt.savefig("../doc/figures/example_opt_ae.pdf", bbox_inches="tight")

