from LCO import LCO_TS, LCO_TS_stencil
import numpy as np
import copy
import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib as matplotlib
import pk_util as pk
import example_ae_setting as dyn_setting
from plot_packages import *

fontsize=30

# optimization path
filename = "opt_ae_hist.dat"
x_path = np.loadtxt(filename)


isLarge = True

ntimeinstance = 5
ndof = 4
if isLarge:
    angle_0 = 5.0
else:
    angle_0 = 0.2
mag_0 = angle_0 / 180.0 * np.pi
if isLarge:
    delta_angle = 4.0
else:
    delta_angle = 0.1
angle_list = [angle_0 - delta_angle, angle_0, angle_0 + delta_angle]
mag_list = copy.deepcopy(angle_list)
for i in range(3):
    mag_list[i] *= np.pi / 180.0
pha_0 = 0.2
ind_p = 1


Nx1 = 20
Nx2 = 20
x1_arr = np.linspace(5.0, 15.0, Nx1)
x2_arr = np.linspace(-3.0, 0.0, Nx2)
stabilty_measure_arr = np.zeros((Nx1, Nx2))
stabilty_measure_FD_arr = np.zeros((Nx1, Nx2))
mu_arr = np.zeros((Nx1, Nx2))
obj = np.zeros((Nx1, Nx2))
# stability_measure_der_adjoint = np.zeros((Nx1, Nx2))
for i in range(Nx1):
    for j in range(Nx2):

        print("i, j", i, j)

        x = [x1_arr[i], x2_arr[j]]

        obj[i, j] = x[0] - x[1] ** 2

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
            oscillator.set_motion_mag_pha(mag, pha_0, ind_p=ind_p)

            # Generate an initial guess using Hopf bifurcation result
            if i == 0 and j == 0:
                Hopf_obj = pk.hopf_bifurcation(x, dyn_setting.get_A)
                mu_crit = Hopf_obj.solve(0.5, 1.4, 1e-4)
                Hopf_obj.compute_eigvec_magpha()
                sol = Hopf_obj.generate_init_sol(mag, pha_0, ntimeinstance, ind_p=ind_p)

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
        oscillator.set_motion_mag_pha(mag_0 + epsilon_actual, pha_0, ind_p=ind_p)

        # Generate an initial guess using Hopf bifurcation result
        Hopf_obj = pk.hopf_bifurcation(x, dyn_setting.get_A)
        mu_crit = Hopf_obj.solve(0.5, 1.4, 1e-4)
        Hopf_obj.compute_eigvec_magpha()
        xin = Hopf_obj.generate_init_sol(mag_0 + epsilon_actual, pha_0, ntimeinstance, ind_p=ind_p)

        # Solve
        sol_perturbed = oscillator.solve(xin)

        stability_measure_FD = (sol_perturbed[0] - mu_midpoint) / epsilon_actual

        print("stability_measure_FD", stability_measure_FD)

        stabilty_measure_FD_arr[i, j] = stability_measure_FD


is_compare_FD = False


fig, ax = plt.subplots(1, 3, figsize=(16, 8))
X1_arr, X2_arr = np.meshgrid(x1_arr, x2_arr)


# ====================
# First plot: stability constraint
# ====================

if isLarge:
    levels0 = np.arange(-0.2, 0.2, 0.02)
else:
    levels0 = np.arange(-0.01, 0.01, 0.001)

CP0 = ax[0].contour(X1_arr, X2_arr, stabilty_measure_arr.T, levels0, extend="both", cmap="coolwarm", linewidths=2, zorder=0)

if isLarge:
    ax[0].clabel(CP0, levels0[1::1], inline=True, fmt="%1.2f", fontsize=fontsize, zorder=0)  # label every second level
else:
    ax[0].clabel(CP0, levels0[1::1], inline=True, fmt="%1.4f", fontsize=fontsize, zorder=0)  # label every second level

ax[0].spines["right"].set_visible(False)
ax[0].spines["top"].set_visible(False)

ax[0].set_xlabel(r"$\bar{m}$", fontsize=fontsize)
ax[0].set_ylabel(r"$\kappa^{(3)}_{\alpha}$", fontsize=fontsize, rotation=0)

ax[0].tick_params(axis='x', labelsize=fontsize)
ax[0].tick_params(axis='y', labelsize=fontsize)

ax[0].yaxis.set_label_coords(-0.4, 0.5)

ax[0].set_xlim(5, 15.5)
ax[0].set_ylim(-3.05,0)

# Extract the stability boundary
ind_critical = 11
print("CP0.collections[ind_critical].get_paths()", CP0.collections[ind_critical].get_paths())
stability_boundary_1 = CP0.collections[ind_critical].get_paths()[0]
stability_boundary_1 = stability_boundary_1.vertices
stability_boundary_2 = CP0.collections[ind_critical].get_paths()[1]
stability_boundary_2 = stability_boundary_2.vertices
stability_boundary = np.concatenate((stability_boundary_1, stability_boundary_2))
# stability_boundary = CP0.collections[ind_critical].get_paths()[0]
# stability_boundary = stability_boundary.vertices

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

# ====================
# Second plot: speed constraint
# ====================

if is_compare_FD:

    CP1 = ax[1].contour(X1_arr, X2_arr, stabilty_measure_FD_arr.T, levels0, extend="both", linewidths=2, zorder=0)

    ax[1].clabel(CP1, levels0[1::1], inline=True, fmt="%1.2f", fontsize=fontsize, zorder=0)  # label every second level
    plt.show()

else:
    levels1 = np.arange(0.6, 1.0, 0.1)

    CP1 = ax[1].contour(X1_arr, X2_arr, mu_arr.T, levels1, extend="both", linewidths=2, zorder=0)

    ax[1].clabel(CP1, levels1[1::1], inline=True, fmt="%1.2f", fontsize=fontsize, zorder=0)  # label every second level

ax[1].spines["right"].set_visible(False)
ax[1].spines["top"].set_visible(False)

ax[1].set_xlabel(r"$\bar{m}$", fontsize=fontsize)
ax[1].set_ylabel(r"$\kappa^{(3)}_{\alpha}$", fontsize=fontsize, rotation=0)

ax[1].tick_params(axis='x', labelsize=fontsize)
ax[1].tick_params(axis='y', labelsize=fontsize)

ax[1].get_yaxis().set_visible(False)

ax[1].set_xlim(5, 15.5)
ax[1].set_ylim(-3.05,0)

# Extract the speed boundary
ind_critical = 2
print("CP1.collections[ind_critical].get_paths()", CP1.collections[ind_critical].get_paths())
speed_boundary_1 = CP1.collections[ind_critical].get_paths()[0]
speed_boundary_1 = speed_boundary_1.vertices
speed_boundary_2 = CP1.collections[ind_critical].get_paths()[1]
speed_boundary_2 = speed_boundary_2.vertices
speed_boundary = np.concatenate((speed_boundary_1, speed_boundary_2))

# speed_boundary = CP1.collections[ind_critical].get_paths()[0]
# speed_boundary = speed_boundary.vertices

# Adding optimization paths
# Optimal solution
# ax[0].plot(x_opt[0], x_opt[1], "o")
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

# ====================
# Third plot: obj
# ====================

levels2 = np.arange(-4.0, 16.0, 2.0)

CP3 = ax[2].contour(X1_arr, X2_arr, obj.T, levels2, extend="both", linewidths=2,zorder=0)

ax[2].clabel(CP3, levels2[1::1], inline=True, fmt="%1.1f", fontsize=fontsize, zorder=0)  #

ax[2].spines["right"].set_visible(False)
ax[2].spines["top"].set_visible(False)

ax[2].set_xlabel(r"$\bar{m}$", fontsize=fontsize)
ax[2].set_ylabel(r"$\kappa^{(3)}_{\alpha}$", fontsize=fontsize, rotation=0)

# Stability boundary
ax[2].plot(stability_boundary[:, 0], stability_boundary[:, 1], "-", color="k", alpha=0.6, zorder=0)
ax[2].fill_between(stability_boundary[:, 0], stability_boundary[:, 1], y2=-3, color=my_purple, alpha=0.3, zorder=0)
ax[2].plot(speed_boundary[:, 0], speed_boundary[:, 1], "-", color="k", alpha=0.6, zorder=0)
ax[2].fill_betweenx(speed_boundary[:, 1], speed_boundary[:, 0], x2=5, color=my_blue, alpha=0.3, zorder=0)

ax[2].tick_params(axis='x', labelsize=fontsize)
ax[2].tick_params(axis='y', labelsize=fontsize)

ax[2].set_ylim(-3.05,0)
ax[2].set_xlim(5, 15.5)

ax[2].get_yaxis().set_visible(False)

# Adding optimization paths
# Optimal solution
# ax[0].plot(x_opt[0], x_opt[1], "o")
# Path
ax[2].plot(x_path[:, 0], x_path[:, 1], "o", color="w", markersize=10, zorder=3)
ax[2].plot(x_path[:, 0], x_path[:, 1], "o", color=my_brown, markersize=6, zorder=3)
# Add arrow to the path
ax[2].quiver(
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
ax[0].fill_between(stability_boundary[:, 0], stability_boundary[:, 1], y2=-3, color=my_purple, alpha=0.3, zorder=0)
ax[0].plot(speed_boundary[:, 0], speed_boundary[:, 1], "-", color="k", alpha=0.6, zorder=0)
ax[0].fill_betweenx(speed_boundary[:, 1], speed_boundary[:, 0], x2=5, color=my_blue, alpha=0.3, zorder=0)


# Stability boundary
ax[1].plot(stability_boundary[:, 0], stability_boundary[:, 1], "-", color="k", alpha=0.6, zorder=0)
ax[1].fill_between(stability_boundary[:, 0], stability_boundary[:, 1], y2=-3, color=my_purple, alpha=0.3, zorder=0)
ax[1].plot(speed_boundary[:, 0], speed_boundary[:, 1], "-", color="k", alpha=0.6, zorder=0)
ax[1].fill_betweenx(speed_boundary[:, 1], speed_boundary[:, 0], x2=5, color=my_blue, alpha=0.3, zorder=0)

ax[0].plot([x_path[0, 0], x_path[0, 0]], [x_path[0, 1], x_path[0, 1]], "s",color="w",  markersize=10, zorder=3)
ax[0].plot([x_path[0, 0], x_path[0, 0]], [x_path[0, 1], x_path[0, 1]], "s",color=my_brown, markersize=6, zorder=3)

ax[0].plot([x_path[-1, 0], x_path[-1, 0]], [x_path[-1, 1], x_path[-1, 1]], "d",color="w",  markersize=15, zorder=3)
ax[0].plot([x_path[-1, 0], x_path[-1, 0]], [x_path[-1, 1], x_path[-1, 1]], "d",color=my_brown, markersize=8, zorder=3)

ax[1].plot([x_path[0, 0], x_path[0, 0]], [x_path[0, 1], x_path[0, 1]], "s",color="w",  markersize=10, zorder=3)
ax[1].plot([x_path[0, 0], x_path[0, 0]], [x_path[0, 1], x_path[0, 1]], "s",color=my_brown, markersize=6, zorder=3)

ax[1].plot([x_path[-1, 0], x_path[-1, 0]], [x_path[-1, 1], x_path[-1, 1]], "d",color="w",  markersize=15, zorder=3)
ax[1].plot([x_path[-1, 0], x_path[-1, 0]], [x_path[-1, 1], x_path[-1, 1]], "d",color=my_brown, markersize=8, zorder=3)

ax[2].plot([x_path[0, 0], x_path[0, 0]], [x_path[0, 1], x_path[0, 1]], "s",color="w",  markersize=10, zorder=3)
ax[2].plot([x_path[0, 0], x_path[0, 0]], [x_path[0, 1], x_path[0, 1]], "s",color=my_brown, markersize=6, zorder=3)

ax[2].plot([x_path[-1, 0], x_path[-1, 0]], [x_path[-1, 1], x_path[-1, 1]], "d",color="w",  markersize=15, zorder=3)
ax[2].plot([x_path[-1, 0], x_path[-1, 0]], [x_path[-1, 1], x_path[-1, 1]], "d",color=my_brown, markersize=8, zorder=3)


plt.savefig("../doc/figures/example_swept_ae.pdf", bbox_inches="tight")

