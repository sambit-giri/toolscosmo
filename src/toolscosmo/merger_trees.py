import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd
from .scipy_func import *

class MergerTree:
    """
    Generates a dark matter halo merger tree based on the extended Press-Schechter
    theory as described by Parkinson et al. (2008), MNRAS, 383, 557.

    This class is a Python port of the Julia code `MergerTreeCosmo.jl` by
    Aniket Nath, which was based on code by Olof Nebrin.

    Attributes:
        M0 (float): Final halo mass at z0 in solar masses.
        z0 (float): Final redshift.
        z_max (float): The maximum redshift to trace the main progenitor back to.
        M_res (float): The minimum mass resolution for progenitors.
        main_branch_history (pd.DataFrame): DataFrame containing the history of the main
                                            progenitor branch after a run.
        subhalo_history (pd.DataFrame): DataFrame containing all subhalos that were
                                        accreted onto the main branch after a run.
    """

    def __init__(self, M0, z0, z_max, M_res):
        """Initializes the MergerTree with halo parameters."""
        self.M0 = M0
        self.z0 = z0
        self.z_max = z_max
        self.M_res = M_res
        
        # Initialize cosmological and algorithm parameters
        self._set_default_params()
        
        # Initialize attributes to store results
        self.main_branch_history = None
        self.subhalo_history = None
        
        # Create the interpolation function for speed, mirroring the Julia code's optimization
        self._create_interpolation_function()

    def _set_default_params(self):
        """Sets the default cosmological and algorithm parameters."""
        # Physical constants (cgs)
        self.G = 6.673e-8
        self.Msun = 1.989e33
        self.pc = 3.086e18
        self.mpc = 1e6 * self.pc
        
        # Default ΛCDM Cosmology
        self.h = 0.674
        self.H0 = self.h * 100e5 / self.mpc  # s^-1
        self.Omega_M0 = 0.315
        self.Omega_L0 = 0.685
        self.delta_crit = 1.686
        self.sigma8_obs = 0.811
        
        # Parkinson et al. (2008) model parameters
        self.G0 = 0.57
        self.gamma1 = 0.38
        self.gamma2 = -0.01
        
        # Redshift step size parameters
        self.epsilon1 = 0.1
        self.epsilon2 = 0.1

    def _create_interpolation_function(self):
        """
        Creates a fast 1D interpolation for the integral J(u_res), which is a 
        major performance optimization.
        """
        print("Creating interpolation function for performance optimization...")
        
        def J_exact_integrand(u):
            # The integrand can be unstable at u=0, so integrate from a small positive number
            if u == 0: return 0
            return (1 + 1/u**2)**(self.gamma1 / 2)

        # Generate x-values and compute the integral at each point
        x_vals = np.logspace(-5, 3, num=600)
        y_vals = np.array([quad(J_exact_integrand, 1e-9, x)[0] for x in x_vals])
        
        # Create the interpolation function and store it
        self.J_interp = interp1d(x_vals, y_vals, bounds_error=False, fill_value="extrapolate")
        print("Done!")

    # --- Core Physical and Cosmological Functions ---

    def sigma(self, M):
        """RMS mass fluctuation as a function of halo mass at z=0."""
        M_eq = 2.4e17 * ((self.Omega_M0 * self.h**2) / 0.14)**(-0.5)
        m = 8 * M / M_eq
        # Add a small epsilon to avoid log(0) for m=1
        m = np.where(m == 1.0, m + 1e-9, m)
        N = 0.0845 * self.sigma8_obs
        with np.errstate(invalid='ignore'): # Ignore warnings for log of negative numbers
             log_term = np.log(m / (1 + m))**3
        return N * np.sqrt((36 / (1 + 3 * m)) - log_term)


    def growth_factor(self, z):
        """Unnormalized growth factor D(z)."""
        Omega_M_z = self.Omega_M0 * (1 + z)**3
        Omega_L_z = self.Omega_L0
        Omega_tot = Omega_M_z + Omega_L_z
        
        Om = Omega_M_z / Omega_tot
        Ol = Omega_L_z / Omega_tot
        
        factor1 = (5.0 * Om) / (2.0 * (1 + z))
        factor2 = Om**(4/7) - Ol + (1 + 0.5 * Om) * (1 + Ol / 70)
        return factor1 / factor2

    def D(self, z):
        """Normalized growth factor."""
        return self.growth_factor(z) / self.growth_factor(0.0)

    def delta(self, z):
        return self.delta_crit / self.D(z)

    def ddelta_dz(self, z):
        """Derivative of delta(z) w.r.t. z."""
        dz = 1e-5
        return (self.delta(z + dz) - self.delta(z - dz)) / (2 * dz)

    def alpha(self, M):
        """Logarithmic derivative of sigma(M): -d(lnσ)/d(lnM)."""
        dM = 1e-6 * M
        d_sigma_dM = (self.sigma(M + dM) - self.sigma(M - dM)) / (2 * dM)
        return -(M / self.sigma(M)) * d_sigma_dM

    # --- Parkinson et al. (2008) Algorithm Functions ---
    
    def _get_S(self, q, z, M2, q_res_params):
        """The function S(q) needed for dN/dq."""
        eta, mu, B_val, sigma2, sigmah, delta_z_val = q_res_params
        
        F1 = q**(eta - 1)
        F2 = (delta_z_val / sigma2)**self.gamma2
        F3 = (sigmah / sigma2)**self.gamma1
        
        return np.sqrt(2/np.pi) * B_val * self.alpha(M2/2) * F1 * \
               (self.G0 / 2**(mu * self.gamma1)) * F2 * F3 * self.ddelta_dz(z)
               
    def _get_R(self, q, M2, q_res_params):
        """The function R(q), also needed for dN/dq."""
        eta, mu, B_val, sigma2, _, _ = q_res_params
        
        sigma1_val = self.sigma(q * M2)
        beta_val = self.beta(eta, M2, q)
        
        # Handle potential division by zero or invalid values
        V_q_num = sigma1_val**2
        V_q_den = (sigma1_val**2 - sigma2**2)
        if V_q_den <= 0: return 0 # Avoid math domain error
        V_q = V_q_num / V_q_den**1.5

        return (self.alpha(q * M2) / self.alpha(M2/2)) * \
               (V_q / (B_val * (q**beta_val))) * \
               (((2 * q)**mu * sigma1_val / self.sigma(M2/2))**self.gamma1)

    def beta(self, eta, M2, q_res):
        return eta + 1 + self.gamma1 * self.alpha(M2/2)
               
    def _get_delta_z(self, z, M2, q_res_params):
        """The redshift step size Δz."""
        eta, _, _, sigma2, sigmah, _ = q_res_params
        S_at_1 = self._get_S(1.0, z, M2, q_res_params)
        
        # Avoid division by zero if eta is close to zero
        if abs(eta) < 1e-9: 
            integral = S_at_1 * np.log(0.5 / (self.M_res/M2))
        else:
            integral = (S_at_1 / eta) * (0.5**eta - (self.M_res/M2)**eta)
        
        dz_min_part = np.sqrt(2) * np.sqrt(sigmah**2 - sigma2**2) / self.ddelta_dz(z)
        
        # Ensure integral is positive before division
        if integral <= 0: return self.epsilon1 * dz_min_part
        
        return min(self.epsilon1 * dz_min_part, self.epsilon2 / integral)
        
    def _get_N_upper(self, z, M2, delta_z, q_res_params):
        """Upper limit on the expected number of fragments."""
        eta, _, _, _, _, _ = q_res_params
        S_at_1 = self._get_S(1.0, z, M2, q_res_params)
        
        if abs(eta) < 1e-9:
             return S_at_1 * np.log(0.5 / (self.M_res/M2)) * delta_z
        return (S_at_1 / eta) * (0.5**eta - (self.M_res/M2)**eta) * delta_z

    def _get_F(self, z, M2, delta_z, q_res_params):
        """Fraction of mass in progenitors below the resolution limit."""
        _, _, _, sigma2, _, delta_z_val = q_res_params
        M_res_val = self.M_res
        sigma_res = self.sigma(M_res_val)
        
        # Avoid math domain error if sigma_res is not larger than sigma2
        if sigma_res**2 <= sigma2**2: return 0.0
        
        u_res = sigma2 / np.sqrt(sigma_res**2 - sigma2**2)
        
        return np.sqrt(2/np.pi) * self.J_interp(u_res) * (self.G0 / sigma2) * \
               ((delta_z_val / sigma2)**self.gamma2) * self.ddelta_dz(z) * delta_z

    # --- Main Execution and Plotting ---

    def run(self, verbose=True):
        """
        Executes the main algorithm to generate the merger tree. The results
        are stored in the instance attributes `main_branch_history` and `subhalo_history`.

        Args:
            verbose (bool): If True, prints progress during iteration.
        """
        # Initialize data storage
        M_main_hist = [self.M0]
        z_main_hist = [self.z0]
        M_sub_hist, z_sub_hist = [], []

        if verbose:
            print("\nStarting iteration...")
            print(f"{'Main Progenitor Mass (M_sun)':<30} {'Redshift (z)':<20}")
            print("-" * 50)

        start_time = time.time()
        
        # Main loop: trace progenitor back in time
        while z_main_hist[-1] < self.z_max and M_main_hist[-1] > 2 * self.M_res:
            M2 = M_main_hist[-1]
            z2 = z_main_hist[-1]
            q_res = self.M_res / M2
            
            if verbose:
                elapsed = (time.time() - start_time) / 60
                print(f"{M2:<30.3e} {z2:<20.6f} (t={elapsed:.2f} mins)")
                
            # Pre-calculate values for this step
            sigma2 = self.sigma(M2)
            sigmah = self.sigma(M2/2)
            delta_z_val = self.delta(z2)
            mu = self.alpha(M2/2)
            
            # Handle cases where sigma values might be NaN or Inf
            if not np.all(np.isfinite([sigma2, sigmah, mu])):
                 if verbose: print("Warning: Non-finite sigma/alpha value encountered. Stopping.")
                 break
            
            V_qres_num = self.sigma(q_res*M2)**2
            V_qres_den = V_qres_num - sigma2**2
            if V_qres_den <= 0: 
                if verbose: print("Warning: V(q_res) denominator non-positive. Stopping.")
                break
            V_qres = V_qres_num / (V_qres_den**1.5)

            V_half_den = sigmah**2 - sigma2**2
            if V_half_den <= 0:
                if verbose: print("Warning: V(1/2) denominator non-positive. Stopping.")
                break
            V_half = (sigmah**2) / (V_half_den**1.5)
            
            beta_val = np.log(V_qres / V_half) / np.log(2*q_res)
            eta = beta_val - 1 - self.gamma1 * mu
            B_val = V_qres / (q_res**beta_val)
            
            q_res_params = (eta, mu, B_val, sigma2, sigmah, delta_z_val)

            delta_z = self._get_delta_z(z2, M2, q_res_params)
            N_up = self._get_N_upper(z2, M2, delta_z, q_res_params)
            F_mass_loss = self._get_F(z2, M2, delta_z, q_res_params)

            r1 = np.random.rand()

            if r1 > N_up:
                M_new = M2 * (1 - F_mass_loss)
                M_main_hist.append(M_new)
                z_main_hist.append(z2 + delta_z)
            else:
                r2 = np.random.rand()
                q = (q_res**eta + (0.5**eta - q_res**eta) * r2)**(1/eta)
                
                r3 = np.random.rand()
                if r3 < self._get_R(q, M2, q_res_params):
                    M_draw1 = M2 * q
                    M_draw2 = M2 * (1 - F_mass_loss - q)
                    
                    M_larger = max(M_draw1, M_draw2)
                    M_smaller = min(M_draw1, M_draw2)
                    
                    M_main_hist.append(M_larger)
                    M_sub_hist.append(M_smaller)
                    z_sub_hist.append(z2 + delta_z)
                else:
                    M_larger = M2 * (1 - F_mass_loss)
                    M_main_hist.append(M_larger)
                
                z_main_hist.append(z2 + delta_z)

        if verbose:
            print("\nIteration complete.")

        # Store results as instance attributes
        self.main_branch_history = pd.DataFrame({
            'mass_msun': M_main_hist,
            'redshift': z_main_hist
        })
        self.subhalo_history = pd.DataFrame({
            'mass_msun': M_sub_hist,
            'redshift': z_sub_hist
        })
        
        return self.main_branch_history, self.subhalo_history

    def plot_merger_history(self, ax=None, save_path=None):
        """
        Plots the generated merger history.

        Args:
            ax (matplotlib.axes.Axes, optional): An existing axes object to plot on.
                                                 If None, a new figure and axes are created.
            save_path (str, optional): Path to save the figure to. If None, the plot is shown.
        """
        if self.main_branch_history is None:
            raise RuntimeError("You must run the simulation with .run() before plotting.")

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 7))
            show_plot = True
        else:
            fig = ax.get_figure()
            show_plot = False

        # Plot the main progenitor branch
        ax.plot(self.main_branch_history['redshift'], self.main_branch_history['mass_msun'], 
                color='royalblue', lw=2.5, label='Main Progenitor')

        # Plot the accreted subhalos
        if not self.subhalo_history.empty:
            ax.scatter(self.subhalo_history['redshift'], self.subhalo_history['mass_msun'],
                       color='crimson', s=50, zorder=5, alpha=0.8, label='Accreted Subhalos')

        # --- Formatting ---
        ax.set_yscale('log')
        ax.set_xlabel('Redshift (z)', fontsize=14)
        ax.set_ylabel(r'Halo Mass ($M_{\odot}$)', fontsize=14)
        ax.set_title(f'Merger History for a ${self.M0:.0e}\, M_{{\odot}}$ Halo', fontsize=16)
        
        # Reverse x-axis to show time evolution from right to left
        ax.invert_xaxis()
        
        ax.grid(True, which='major', linestyle='--', linewidth=0.5)
        ax.legend(fontsize=12)
        ax.tick_params(axis='both', which='major', labelsize=12)
        
        fig.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300)
            print(f"Plot saved to {save_path}")
        
        if show_plot:
            plt.show()
