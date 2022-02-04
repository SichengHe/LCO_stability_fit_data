from LCO import LCO_TS, LCO_TS_stencil
import numpy as np
import copy
import pk_util as pk

isSubcritical = False

def force(mu, w, x):

    '''
        Forcing term.
    '''

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
    MsInv_Ks[0, 1] = coeff * (- xa_con * ra_con ** 2)
    MsInv_Ks[1, 0] = coeff * (- xa_con * Omega_con ** 2)
    MsInv_Ks[1, 1] = coeff * ra_con ** 2

    # Ms^{-1} * Ka
    MsInv_Ka = np.zeros((2, 2))
    coeff = 2.0 * mu ** 2 / ((ra_con ** 2 - xa_con ** 2) * mbar)
    MsInv_Ka[0, 1] = coeff * (ra_con ** 2 + xa_con * ebar_con)
    MsInv_Ka[1, 1] = coeff * (- xa_con - ebar_con)

    # Ms^{-1} * Da
    MsInv_Da = np.zeros((2, 2))
    coeff = 2.0 * mu / ((ra_con ** 2 - xa_con ** 2) * mbar)
    MsInv_Da[0, 0] = coeff * (ra_con ** 2 + xa_con * ebar_con)
    MsInv_Da[1, 0] = coeff * (- xa_con - ebar_con)

    A[2:4, 0:2] += - 1.0 * MsInv_Ks[:, :]
    A[2:4, 0:2] += - 1.0 * MsInv_Ka[:, :]
    A[2:4, 2:4] += - 1.0 * MsInv_Da[:, :]

    # Nonlinear force
    Fnl = np.zeros(4)
    coeff = (ra_con ** 2 * (kappa_3 * alpha ** 3 + kappa_5_con * alpha ** 5)) / (ra_con ** 2 - xa_con ** 2)
    Fnl[2] = coeff * xa_con
    Fnl[3] = coeff * ( -1.0 )

    # Total force
    f = np.zeros(4)
    f[:] += A.dot(w) + Fnl[:]

    return f

def pforcepw(mu, w, x):

    '''
        pf / pw
    '''

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
    MsInv_Ks[0, 1] = coeff * (- xa_con * ra_con ** 2)
    MsInv_Ks[1, 0] = coeff * (- xa_con * Omega_con ** 2)
    MsInv_Ks[1, 1] = coeff * ra_con ** 2

    # Ms^{-1} * Ka
    MsInv_Ka = np.zeros((2, 2))
    coeff = 2.0 * mu ** 2 / ((ra_con ** 2 - xa_con ** 2) * mbar)
    MsInv_Ka[0, 1] = coeff * (ra_con ** 2 + xa_con * ebar_con)
    MsInv_Ka[1, 1] = coeff * (- xa_con - ebar_con)

    # Ms^{-1} * Da
    MsInv_Da = np.zeros((2, 2))
    coeff = 2.0 * mu / ((ra_con ** 2 - xa_con ** 2) * mbar)
    MsInv_Da[0, 0] = coeff * (ra_con ** 2 + xa_con * ebar_con)
    MsInv_Da[1, 0] = coeff * (- xa_con - ebar_con)

    A[2:4, 0:2] += - 1.0 * MsInv_Ks[:, :]
    A[2:4, 0:2] += - 1.0 * MsInv_Ka[:, :]
    A[2:4, 2:4] += - 1.0 * MsInv_Da[:, :]

    # Nonlinear force
    pFnlpw = np.zeros((4, 4))
    coeff = (ra_con ** 2 * (kappa_3 * 3.0 * alpha ** 2 + kappa_5_con * 5.0 * alpha ** 4)) / (ra_con ** 2 - xa_con ** 2)
    pFnlpw[2, 1] = coeff * xa_con
    pFnlpw[3, 1] = coeff * ( -1.0 )

    # Total force
    pfpw = np.zeros((4, 4))
    pfpw[:, :] += A[:, :]
    pfpw[:, :] += pFnlpw[:, :]

    return pfpw

def pforcepx(mu, w, x):

    '''
        p f / p x
    '''

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
    coeff = - 2.0 * mu ** 2 / ((ra_con ** 2 - xa_con ** 2) * mbar ** 2)
    pMsInv_Ka_pmbar[0, 1] = coeff * (ra_con ** 2 + xa_con * ebar_con)
    pMsInv_Ka_pmbar[1, 1] = coeff * (- xa_con - ebar_con)

    # Ms^{-1} * Da
    pMsInv_Da_pmbar = np.zeros((2, 2))
    coeff = - 2.0 * mu / ((ra_con ** 2 - xa_con ** 2) * mbar ** 2)
    pMsInv_Da_pmbar[0, 0] = coeff * (ra_con ** 2 + xa_con * ebar_con)
    pMsInv_Da_pmbar[1, 0] = coeff * (- xa_con - ebar_con)

    pA_pmbar[2:4, 0:2] += - 1.0 * pMsInv_Ka_pmbar[:, :]
    pA_pmbar[2:4, 2:4] += - 1.0 * pMsInv_Da_pmbar[:, :]

    # Nonlinear force
    pFnl_pkappa_3 = np.zeros(4)
    coeff = (ra_con ** 2 * ( alpha ** 3 )) / (ra_con ** 2 - xa_con ** 2)
    pFnl_pkappa_3[2] = coeff * xa_con
    pFnl_pkappa_3[3] = coeff * ( -1.0 )

    # Total force
    pfpx = np.zeros((4, 2))
    # pf / pmbar
    pfpx[:, 0] += pA_pmbar.dot(w)[:]
    # pf / p kappa_3
    pfpx[:, 1] += pFnl_pkappa_3[:]

    return pfpx

def pforcepmu(mu, w, x):

    '''
        p f / p mu
    '''

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
    pMsinv_Ka_pmu[1, 1] = coeff * (- xa_con - ebar_con)

    # Ms^{-1} * Da
    pMsinv_Da_pmu = np.zeros((2, 2))
    coeff = 2.0 / ((ra_con ** 2 - xa_con ** 2) * mbar)
    pMsinv_Da_pmu[0, 0] = coeff * (ra_con ** 2 + xa_con * ebar_con)
    pMsinv_Da_pmu[1, 0] = coeff * (- xa_con - ebar_con)

    pA_pmu[2:4, 0:2] += - 1.0 * pMsinv_Ka_pmu[:, :]
    pA_pmu[2:4, 2:4] += - 1.0 * pMsinv_Da_pmu[:, :]

    # Total force
    pfpmu = np.zeros(4)
    pfpmu[:] += pA_pmu.dot(w)[:]

    return pfpmu

def get_A(mu, x):

    '''
        Forcing term.
    '''

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
    MsInv_Ks[0, 1] = coeff * (- xa_con * ra_con ** 2)
    MsInv_Ks[1, 0] = coeff * (- xa_con * Omega_con ** 2)
    MsInv_Ks[1, 1] = coeff * ra_con ** 2

    # Ms^{-1} * Ka
    MsInv_Ka = np.zeros((2, 2))
    coeff = 2.0 * mu ** 2 / ((ra_con ** 2 - xa_con ** 2) * mbar)
    MsInv_Ka[0, 1] = coeff * (ra_con ** 2 + xa_con * ebar_con)
    MsInv_Ka[1, 1] = coeff * (- xa_con - ebar_con)

    # Ms^{-1} * Da
    MsInv_Da = np.zeros((2, 2))
    coeff = 2.0 * mu / ((ra_con ** 2 - xa_con ** 2) * mbar)
    MsInv_Da[0, 0] = coeff * (ra_con ** 2 + xa_con * ebar_con)
    MsInv_Da[1, 0] = coeff * (- xa_con - ebar_con)

    A[2:4, 0:2] += - 1.0 * MsInv_Ks[:, :]
    A[2:4, 0:2] += - 1.0 * MsInv_Ka[:, :]
    A[2:4, 2:4] += - 1.0 * MsInv_Da[:, :]

    return A

isBasicCheck = False
if isBasicCheck:
    ndof = 4
    w = np.random.rand( ndof )
    x = np.random.rand( 2 )
    nx = len(x)
    mu = np.random.rand()

    force_0 = force(mu, w, x)

    epsilon = 1e-7

    # p force/ p w
    pforcepw_FD = np.zeros((ndof, ndof))
    for j in range(ndof):
        w_perturbed = copy.deepcopy(w)
        w_perturbed[j] += epsilon
        force_perturbed = force(mu, w_perturbed, x)

        pforcepw_loc = (force_perturbed - force_0) / epsilon

        pforcepw_FD[:, j] = pforcepw_loc[:]

    pforcepw_AD = pforcepw(mu, w, x)

    print("pforcepw_AD", pforcepw_AD)
    print("pforcepw_FD", pforcepw_FD)

    # p R / p x
    pforcepx_FD = np.zeros((ndof, nx))
    for j in range(nx):
        x_perturbed = copy.deepcopy(x)
        x_perturbed[j] += epsilon
        force_perturbed = force(mu, w, x_perturbed)

        pforcepx_loc = (force_perturbed - force_0) / epsilon

        pforcepx_FD[:, j] = pforcepx_loc[:]

    pforcepx_AD = pforcepx(mu, w, x)

    print("pforcepx_AD", pforcepx_AD)
    print("pforcepx_FD", pforcepx_FD)

    # p R / p mu
    pforcepmu_FD = np.zeros(ndof)
    mu_perturbed = mu + epsilon
    force_perturbed = force(mu_perturbed, w, x)

    pforcepmu_FD = (force_perturbed - force_0) / epsilon

    pforcepmu_AD = pforcepmu(mu, w, x)

    print("pforcepmu_AD", pforcepmu_AD)
    print("pforcepmu_FD", pforcepmu_FD)

else:

    ntimeinstance = 5
    ndof = 4
    angle_0 = 5.0
    mag_0 = angle_0 / 180.0 * np.pi
    pha_0 = 0.2
    if isSubcritical:
        x0 = [10.0, - 1.5]
    else:
        x0 = [10.0, 1.5]
    nx = len(x0)
    delta_angle = 4.0
    angle_list = [angle_0 - delta_angle, \
    angle_0, \
    angle_0 + delta_angle]
    mag_list = copy.deepcopy(angle_list)
    for i in range(3):
        mag_list[i] *= (np.pi / 180.0)
    ind_p = 1

    oscillator_list = []
    sol_list = []
    for i in range(3):
        mag = mag_list[i]
        oscillator = LCO_TS(force, ntimeinstance, ndof, x = x0, \
            pforcepx_func = pforcepx, pforcepw_func = pforcepw, \
            pforcepmu_func = pforcepmu)
        oscillator.set_motion_mag_pha(mag, pha_0, ind_p = ind_p)

        # Generate an initial guess using Hopf bifurcation result
        Hopf_obj = pk.hopf_bifurcation(x0, get_A)
        mu_crit = Hopf_obj.solve(0.5, 1.4, 1e-4)
        Hopf_obj.compute_eigvec_magpha()
        xin = Hopf_obj.generate_init_sol(mag, pha_0, ntimeinstance, ind_p = ind_p)

        sol = oscillator.solve(xin)
        print("sol", sol)

        oscillator_list.append(oscillator)
        sol_list.append(sol)

    # Ajoint fitted
    LCO_stability = LCO_TS_stencil(oscillator_list, mag_list)
    LCO_stability.compute_stability()
    stability_measure_0 = LCO_stability.get_stability()
    LCO_stability.compute_stability_der()
    stability_measure_der_adjoint = LCO_stability.get_stability_der()

    # FD fitted
    stability_measure_der_FD = np.zeros(nx)
    epsilon = 1e-8
    for i in range(nx):
        x0_per = copy.deepcopy(x0)
        x0_per[i] += epsilon

        LCO_stability.set_design_var(x0_per)
        LCO_stability.solve(sol_list)
        LCO_stability.reset_mu()

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
        oscillator.set_motion_mag_pha(mag, pha_0, ind_p = ind_p)

        # Generate an initial guess using Hopf bifurcation result
        Hopf_obj = pk.hopf_bifurcation(x0, get_A)
        mu_crit = Hopf_obj.solve(0.5, 1.4, 1e-4)
        Hopf_obj.compute_eigvec_magpha()
        xin = Hopf_obj.generate_init_sol(mag, pha_0, ntimeinstance, ind_p = ind_p)

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


