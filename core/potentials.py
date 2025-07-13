# core/potentials.py
import numpy as np
from scipy.optimize import minimize

class FHRGPotential:
    def __init__(self, gamma0=1.45, lambda_val=5e-49, kappa=0.9, 
                 alpha=-2.42, Delta=3.1, rho_Lambda=2.5e-47):
        self.gamma0 = gamma0
        self.lambda_val = lambda_val
        self.kappa = kappa
        self.alpha = alpha
        self.Delta = Delta
        self.rho_Lambda = rho_Lambda
        self.mu4 = self.calculate_mu4()
        self.V0 = self.calculate_V0()
    
    def calculate_mu4(self):
        """Calculate μ⁴ from dark energy density"""
        return self.rho_Lambda * np.exp(self.kappa * self.gamma0)
    
    def calculate_V0(self):
        """Calculate offset potential V₀"""
        term = self.mu4 * np.exp(-self.kappa * self.gamma0) * \
               (1 + self.alpha * self.gamma0**(self.Delta-4))
        return self.rho_Lambda - term
    
    def value(self, gamma):
        """Full potential V(γ)"""
        bulk_term = self.mu4 * np.exp(-self.kappa * gamma) * \
                    (1 + self.alpha * gamma**(self.Delta-4))
        stab_term = 0.25 * self.lambda_val * (gamma - self.gamma0)**4
        return bulk_term + stab_term + self.V0
    
    def first_derivative(self, gamma):
        """dV/dγ"""
        exp_term = np.exp(-self.kappa * gamma)
        poly_term = 1 + self.alpha * gamma**(self.Delta-4)
        deriv_poly = self.alpha * (self.Delta-4) * gamma**(self.Delta-5)
        
        term1 = self.mu4 * exp_term * (-self.kappa * poly_term + deriv_poly)
        term2 = self.lambda_val * (gamma - self.gamma0)**3
        return term1 + term2
    
    def second_derivative(self, gamma):
        """d²V/dγ²"""
        h = 1e-6
        return (self.first_derivative(gamma + h) - self.first_derivative(gamma - h)) / (2*h)
    
    def mass_squared(self):
        """m_γ² at γ₀"""
        return self.second_derivative(self.gamma0)
    
    def find_minimum(self):
        """Numerically find γ₀"""
        res = minimize(self.value, x0=1.5, method='L-BFGS-B')
        return res.x[0]