import numpy as np
import cmath
import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib as matplotlib

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


class hopf_bifurcation(object):

    """
        Hopf bifurcation class. 
    """

    def __init__(self, x, func_A, ndof=4):

        """
            Initiate with design var and coeff function.
        """

        # Design variable
        self.x = x

        # Coeff function
        # dot{w} = A w
        # A = A(mu, x)
        self.func_A = func_A

        self.ndof = ndof

    def compute_max_eig(self, mu):

        """
            Extract the eigenvalue with the maximum real part.
        """

        A = self.func_A(mu, self.x)

        eig_arr = np.linalg.eig(A)[0]
        eig_real_arr = np.real(eig_arr)

        return np.max(eig_real_arr)

    def solve(self, muL, muR, delta):

        """
            Solve for f(mu) = 0 using bisection.
            It is assumed that f(muL) * f(muR) < 0.
            And a solution of precision delta is searched for.
        """

        if muR <= muL:
            print("Error: Right bound less or equal to the left.")
            exit()

        # Evaluate function value
        f_muR = self.compute_max_eig(muR)
        f_muL = self.compute_max_eig(muL)

        # Safety check
        if f_muR * f_muL > 0:
            print("Error: Not sure solution in between.")
            exit()
        elif f_muR == 0:
            return muR
        elif f_muL == 0:
            return muL

        # Main loop for the search
        while muR - muL > delta:

            mu_new = (muR + muL) / 2
            f_mu_new = self.compute_max_eig(mu_new)

            if f_mu_new == 0:
                return mu_new
            if f_mu_new * f_muR < 0:
                muL = mu_new
                f_muL = f_mu_new
            elif f_mu_new * f_muL < 0:
                muR = mu_new
                f_muR = f_mu_new
            else:
                print("Error: Something unexpected happened...")

        # Set the solution
        self.mu = (muR + muL) / 2

        return self.mu

    def compute_eigvec_magpha(self):

        """
            Compute the eigenvector phase and magnitude for
            the eigenvector corresponds with the bifurcation
            eigenvalue (real part = 0).
        """

        # Extract the coeff matrix for the bifurcation point.
        A = self.func_A(self.mu, self.x)
        eigval_arr, eigvec_arr = np.linalg.eig(A)

        # Get the bifurcation vector.
        ind = np.argmax(np.real(eigval_arr))
        eigvec = eigvec_arr[:, ind]

        # Construct the magnitude and phase arrays.
        n = len(eigvec)

        self.eigmag = np.zeros(n)
        self.eigpha = np.zeros(n)

        for i in range(n):
            self.eigmag[i] = abs(eigvec[i])
            self.eigpha[i] = cmath.phase(eigvec[i])

        # Renormalize the phase based on the first entry
        eigpha0 = self.eigpha[0]
        for i in range(n):

            self.eigpha[i] -= eigpha0

        return self.eigmag, self.eigpha

    def generate_init_sol(self, mag, pha, N, ind_p=0):

        # Generate the array.
        sol = np.zeros(self.ndof * N + 2)

        # mu
        sol[0] = self.mu

        # omega
        A = self.func_A(self.mu, self.x)
        eigval_arr, eigvec_arr = np.linalg.eig(A)
        ind = np.argmax(np.real(eigval_arr))
        omega = np.imag(eigval_arr[ind])

        sol[1] = omega

        ratio = mag / self.eigmag[ind_p]

        for i in range(N):
            for j in range(self.ndof):

                mag_loc = self.eigmag[j] * ratio
                pha_loc = self.eigpha[j] + pha + (float(i) / float(N)) * (2.0 * np.pi)

                sol[2 + i * self.ndof + j] = mag_loc * np.sin(pha_loc)

        return sol

    def plot(self):

        """
            Plot the eigenvalue sweep for the 4 eigenvalues.
        """

        # Bounds
        muL = 0.0
        muR = 1.4

        # Compute the eigenvalues
        N = 100

        eig_list_1 = []
        eig_list_2 = []
        eig_list_3 = []
        eig_list_4 = []

        mu_arr = np.linspace(muL, muR, N)
        max_mu_arr = np.zeros(N)
        for i in range(N):
            mu = mu_arr[i]

            A = self.func_A(mu, self.x)

            eig = np.linalg.eig(A)[0]

            eig_list_1.append(eig[0])
            eig_list_2.append(eig[1])
            eig_list_3.append(eig[2])
            eig_list_4.append(eig[3])

        # Plot
        fig, (ax1, ax2) = plt.subplots(2, 1)

        markersize = 2

        ax1.plot(mu_arr, np.real(eig_list_1), "o", color=my_blue, markersize=markersize)
        ax1.plot(mu_arr, np.real(eig_list_2), "o", color=my_blue, markersize=markersize)
        ax1.plot(mu_arr, np.real(eig_list_3), "o", color=my_blue, markersize=markersize)
        ax1.plot(mu_arr, np.real(eig_list_4), "o", color=my_blue, markersize=markersize)

        ax1.plot([muL, muR], [0, 0], "k-", alpha=0.3)

        ax1.set_ylabel(r"$\mathrm{Re}(\lambda)$", fontsize=20, rotation=0)

        ax1.spines["right"].set_visible(False)
        ax1.spines["top"].set_visible(False)
        ax1.xaxis.set_visible(False)

        ax1.set_xlim([muL, muR])

        ax2.plot(mu_arr, np.imag(eig_list_1), "o", color=my_blue, markersize=markersize)
        ax2.plot(mu_arr, np.imag(eig_list_2), "o", color=my_blue, markersize=markersize)
        ax2.plot(mu_arr, np.imag(eig_list_3), "o", color=my_blue, markersize=markersize)
        ax2.plot(mu_arr, np.imag(eig_list_4), "o", color=my_blue, markersize=markersize)

        ax2.set_xlabel(r"$\mu$", fontsize=20)
        ax2.set_ylabel(r"$\mathrm{Im}(\lambda)$", fontsize=20, rotation=0)

        ax2.spines["right"].set_visible(False)
        ax2.spines["top"].set_visible(False)

        ax2.set_xlim([muL, muR])

        plt.show()


if 0:

    def get_A(mu, x):

        """
            Forcing term.
        """

        # Define constants
        ebar_con = 0.2
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

    # x = [10.0, -1.5]
    x = [10.0, 1.5]
    Hopf_obj = hopf_bifurcation(x, get_A)
    # Hopf_obj.plot()
    mu_crit = Hopf_obj.solve(0.5, 1.4, 1e-4)
    eigmag, eigpha = Hopf_obj.compute_eigvec_magpha()

    print("mu_crit", mu_crit)
    print("eigmag", eigmag)
    print("eigpha", eigpha)
