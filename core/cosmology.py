# core/cosmology.py
import numpy as np
from scipy.integrate import solve_ivp

class FHRGCosmology:
    def __init__(self, potential, H0=67.4, Omega_m0=0.311):
        self.pot = potential
        self.H0 = H0  # km/s/Mpc
        self.Omega_m0 = Omega_m0
        self.G = 4.3009e-9  # pc M⊙⁻¹ (km/s)²
        
    def friedmann_equations(self, z, state):
        """Modified Friedmann equations with γ-field"""
        gamma, gamma_dot = state
        a = 1/(1+z)
        H = self.H0 * np.sqrt(self.Omega_m0 * a**-3 + 
                             self.Omega_phi(z, state))
        
        dgamma_dz = -gamma_dot / (H * (1+z))
        dgamma_dot_dz = (3*gamma_dot/(1+z) - 
                         self.pot.first_derivative(gamma)/(H*(1+z))) / H
        
        return [dgamma_dz, dgamma_dot_dz]
    
    def Omega_phi(self, z, state):
        """Density parameter for γ-field"""
        gamma, gamma_dot = state
        rho_phi = 0.5 * gamma_dot**2 + self.pot.value(gamma)
        rho_crit = 3 * (self.H0*1e3/3.0856e22)**2 / (8*np.pi*self.G)  # GeV⁴
        return rho_phi / rho_crit
    
    def solve(self, z_span=[0, 10], initial_conditions=[1.45, 0]):
        """Solve cosmological equations"""
        sol = solve_ivp(self.friedmann_equations, [z_span[0], z_span[1]], 
                        initial_conditions, t_eval=np.linspace(*z_span, 1000),
                        method='LSODA')
        return sol
    
    def equation_of_state(self, z, state):
        """w(z) for dark energy"""
        gamma, gamma_dot = state
        kinetic = 0.5 * gamma_dot**2
        potential = self.pot.value(gamma)
        return (kinetic - potential) / (kinetic + potential)
    
    def Hubble_parameter(self, z, state):
        """H(z) in km/s/Mpc"""
        a = 1/(1+z)
        return self.H0 * np.sqrt(self.Omega_m0 * a**-3 + 
                                self.Omega_phi(z, state))
    
    def S8_calculation(self, z_values, gamma_values):
        """Compute S₈ parameter"""
        # Simplified growth factor calculation
        a_values = 1/(1 + z_values)
        gamma0 = self.pot.gamma0
        integrand = 0.1 * (gamma_values - gamma0) / a_values
        
        # Integrate using trapezoidal rule
        integral = np.zeros_like(a_values)
        for i in range(1, len(a_values)):
            integral[i] = integral[i-1] + 0.5 * (integrand[i] + integrand[i-1]) * \
                         (np.log(a_values[i]) - np.log(a_values[i-1]))
        
        # LCDM reference value
        sigma8_LCDM = 0.811 * (a_values[-1])**0.5
        
        return sigma8_LCDM * np.exp(integral) * np.sqrt(self.Omega_m0 / 0.3)