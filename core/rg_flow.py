# core/rg_flow.py
import numpy as np
from scipy.integrate import solve_ivp

class RGFlowSolver:
    def __init__(self, gamma0=1.45, C_dm=0.37, C_grav=-0.032):
        self.gamma0 = gamma0
        self.C_dm = C_dm
        self.C_grav = C_grav
        
    def beta_SM(self, gamma):
        """Standard Model contribution (simplified)"""
        return -0.01 * (gamma - 4)**2
        
    def equation(self, z, gamma):
        """dγ/dz RG equation"""
        return self.C_dm * (1+z)**3 * np.exp(-gamma) + \
               self.C_grav * gamma**2 + \
               self.beta_SM(gamma)
               
    def solve(self, z_span=[0, 10], gamma_init=4.0):
        """Solve RG flow equation"""
        sol = solve_ivp(self.equation, [z_span[0], z_span[1]], [gamma_init],
                        t_eval=np.linspace(*z_span, 1000))
        return sol