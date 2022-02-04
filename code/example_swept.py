from LCO import LCO_TS, LCO_TS_stencil
import numpy as np
import copy
import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib as matplotlib
import pk_util as pk
import example_simple_setting as dyn_setting
from plot_packages import *

fontsize=30

# optimization path
filename = "opt_hist.dat"
x_path = np.loadtxt(filename)


ntimeinstance = 5
ndof = 2
mag_0 = 0.5
pha_0 = 0.2
delta_mag = 0.1
mag_list = [mag_0 - delta_mag, mag_0, mag_0 + delta_mag]


Nx1 = 20
Nx2 = 20
x1_arr = np.linspace(0, 1, Nx1)
x2_arr = np.linspace(0, 1, Nx2)
stabilty_measure_arr = np.zeros((Nx1, Nx2))
stabilty_measure_FD_arr = np.zeros((Nx1, Nx2))
mu_arr = np.zeros((Nx1, Nx2))
# stability_measure_der_adjoint = np.zeros((Nx1, Nx2))
for i in range(Nx1):
    for j in range(Nx2):

        print("i, j", i, j)

        x = [x1_arr[i], x2_arr[j]]

        # Compute the stability der using curve fitting

        oscillator_list = []
        for ii in range(3):
            mag = mag_list[ii]
            oscillator = LCO_TS(
                dyn_setting.force,
                ntimeinstance,
                ndof,
                x=x,
                pforcepx_func=dyn_setting.pforcepx,
                pforcepw_func=dyn_setting.pforcepw,
                pforcepmu_func=dyn_setting.pforcepmu,
            )
            oscillator.set_motion_mag_pha(mag, pha_0)

            # Generate an initial guess using Hopf bifurcation result
            if i == 0 and j == 0:
                Hopf_obj = pk.hopf_bifurcation(x, dyn_setting.get_A, ndof=2)
                mu_crit = Hopf_obj.solve(0.5, 1.4, 1e-4)
                Hopf_obj.compute_eigvec_magpha()
                sol = Hopf_obj.generate_init_sol(mag, pha_0, ntimeinstance)

            sol = oscillator.solve(sol)
            if ii == 1:
                mu_midpoint = sol[0]
                mu_arr[i, j] = mu_midpoint

            oscillator_list.append(oscillator)

        LCO_stability = LCO_TS_stencil(oscillator_list, mag_list)
        LCO_stability.compute_stability()
        stability_measure_0 = LCO_stability.get_stability()

        stabilty_measure_arr[i, j] = stability_measure_0

        # Compute the stability der using FD

        epsilon_actual = 1e-7
        oscillator = LCO_TS(
            dyn_setting.force,
            ntimeinstance,
            ndof,
            x=x,
            pforcepx_func=dyn_setting.pforcepx,
            pforcepw_func=dyn_setting.pforcepw,
            pforcepmu_func=dyn_setting.pforcepmu,
        )
        oscillator.set_motion_mag_pha(mag_0 + epsilon_actual, pha_0)

        # Generate an initial guess using Hopf bifurcation result
        Hopf_obj = pk.hopf_bifurcation(x, dyn_setting.get_A, ndof=2)
        mu_crit = Hopf_obj.solve(0.5, 1.4, 1e-4)
        Hopf_obj.compute_eigvec_magpha()
        xin = Hopf_obj.generate_init_sol(mag_0 + epsilon_actual, pha_0, ntimeinstance)

        # Solve
        sol_perturbed = oscillator.solve(xin)

        stability_measure_FD = (sol_perturbed[0] - mu_midpoint) / epsilon_actual

        print("stability_measure_FD", stability_measure_FD)

        stabilty_measure_FD_arr[i, j] = stability_measure_FD


is_compare_FD = False


fig, ax = plt.subplots(1, 2, figsize=(16, 8))
X1_arr, X2_arr = np.meshgrid(x1_arr, x2_arr)

# ====================
# First plot: constraint
# ====================

levels0 = np.arange(-1.0, 1.0, 0.1)

CP0 = ax[0].contour(X1_arr, X2_arr, stabilty_measure_arr.T, levels0, extend="both", cmap="coolwarm", linewidths=2, zorder=0)

ax[0].clabel(CP0, levels0[1::1], inline=True, fmt="%1.1f", fontsize=fontsize, zorder=0)  # label every second level

ax[0].spines["right"].set_visible(False)
ax[0].spines["top"].set_visible(False)

ax[0].set_xlabel(r"$x_1$", fontsize=fontsize)
ax[0].set_ylabel(r"$x_2$", fontsize=fontsize, rotation=0)

ax[0].tick_params(axis='x', labelsize=fontsize)
ax[0].tick_params(axis='y', labelsize=fontsize)

ax[0].yaxis.set_label_coords(-0.2, 0.5)

ax[0].set_xlim(0, 1.05)
ax[0].set_ylim(0, 1.05)

# Extract the stability boundary
ind_critical = 11
print("CP0.collections[ind_critical].get_paths()", CP0.collections[ind_critical].get_paths())
stability_boundary_1 = CP0.collections[ind_critical].get_paths()[0]
stability_boundary_1 = stability_boundary_1.vertices
stability_boundary_2 = CP0.collections[ind_critical].get_paths()[1]
stability_boundary_2 = stability_boundary_2.vertices
stability_boundary = np.concatenate((stability_boundary_1, stability_boundary_2))

# Adding optimization paths
# Optimal solution
# ax[0].plot(x_opt[0], x_opt[1], "o")
# Path
ax[0].plot(x_path[:, 0], x_path[:, 1], "o", color="w", markersize=10, zorder=3)
ax[0].plot(x_path[:, 0], x_path[:, 1], "o", color=my_brown, markersize=6, zorder=3)
# Add arrow to the path
ax[0].quiver(
    x_path[:-1, 0],
    x_path[:-1, 1],
    x_path[1:, 0] - x_path[:-1, 0],
    x_path[1:, 1] - x_path[:-1, 1],
    color=my_brown,
    scale_units="xy",
    angles="xy",
    scale=1,
    zorder=2,
)

# Stability boundary
ax[0].plot(stability_boundary[:, 0], stability_boundary[:, 1], "-", color="k", alpha=0.6, zorder=0)
ax[0].fill_between(stability_boundary[:, 0], stability_boundary[:, 1], y2=0, color=my_purple, alpha=0.3, zorder=0)

# ====================
# Second plot: obj
# ====================
if is_compare_FD:

    CP1 = ax[1].contour(X1_arr, X2_arr, stabilty_measure_FD_arr.T, levels0, extend="both", linewidths=2, zorder=0)

    ax[1].clabel(CP1, levels0[1::1], inline=True, fmt="%1.1f", fontsize=fontsize,zorder=0)  # label every second level

else:
    levels1 = np.arange(-1.0, 1.0, 0.1)

    CP1 = ax[1].contour(X1_arr, X2_arr, mu_arr.T, levels1, extend="both", linewidths=2, zorder=0)

    ax[1].clabel(CP1, levels1[1::1], inline=True, fmt="%1.1f", fontsize=fontsize, zorder=0)  # label every second level

ax[1].spines["right"].set_visible(False)
ax[1].spines["top"].set_visible(False)

ax[1].set_xlabel(r"$x_1$", fontsize=fontsize)
ax[1].set_ylabel(r"$x_2$", fontsize=fontsize, rotation=0)

ax[1].tick_params(axis='x', labelsize=fontsize)
ax[1].tick_params(axis='y', labelsize=fontsize)

ax[1].get_yaxis().set_visible(False)

# Adding optimization paths
# Optimal solution
# ax[1].plot(x_opt[0], x_opt[1], "o")
# Path
ax[1].plot(x_path[:, 0], x_path[:, 1], "o", color="w", markersize=10, zorder=3)
ax[1].plot(x_path[:, 0], x_path[:, 1], "o", color=my_brown, markersize=6, zorder=3)
# Add arrow to the path
ax[1].quiver(
    x_path[:-1, 0],
    x_path[:-1, 1],
    x_path[1:, 0] - x_path[:-1, 0],
    x_path[1:, 1] - x_path[:-1, 1],
    color=my_brown,
    scale_units="xy",
    angles="xy",
    scale=1,
    zorder=2,
)
# Stability boundary
ax[1].plot(stability_boundary[:, 0], stability_boundary[:, 1], "-", color="k", alpha=0.6, zorder=0)
ax[1].fill_between(stability_boundary[:, 0], stability_boundary[:, 1], y2=0, color=my_purple, alpha=0.3, zorder=0)

ax[1].set_xlim(0, 1.05)
ax[1].set_ylim(0, 1.05)

ax[1].plot(x_path[:, 0], x_path[:, 1], "o", color="w", markersize=10, zorder=3)
ax[1].plot(x_path[:, 0], x_path[:, 1], "o", color=my_brown, markersize=6, zorder=3)

ax[0].plot([x_path[0, 0], x_path[0, 0]], [x_path[0, 1], x_path[0, 1]], "s",color="w",  markersize=10, zorder=3)
ax[0].plot([x_path[0, 0], x_path[0, 0]], [x_path[0, 1], x_path[0, 1]], "s",color=my_brown, markersize=6, zorder=3)

ax[0].plot([x_path[-1, 0], x_path[-1, 0]], [x_path[-1, 1], x_path[-1, 1]], "d",color="w",  markersize=15, zorder=3)
ax[0].plot([x_path[-1, 0], x_path[-1, 0]], [x_path[-1, 1], x_path[-1, 1]], "d",color=my_brown, markersize=8, zorder=3)

ax[1].plot([x_path[0, 0], x_path[0, 0]], [x_path[-1, 1], x_path[-1, 1]], "s",color="w",  markersize=10, zorder=3)
ax[1].plot([x_path[0, 0], x_path[0, 0]], [x_path[-1, 1], x_path[-1, 1]], "s",color=my_brown, markersize=6, zorder=3)

ax[1].plot([x_path[-1, 0], x_path[-1, 0]], [x_path[-1, 1], x_path[-1, 1]], "d",color="w",  markersize=15, zorder=3)
ax[1].plot([x_path[-1, 0], x_path[-1, 0]], [x_path[-1, 1], x_path[-1, 1]], "d",color=my_brown, markersize=8, zorder=3)


plt.tight_layout()
plt.savefig("../doc/figures/example_swept.pdf", bbox_inches="tight")
