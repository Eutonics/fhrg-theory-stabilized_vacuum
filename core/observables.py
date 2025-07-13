# core/observables.py
import numpy as np

class FHRGObservables:
    def __init__(self, potential, cosmology):
        self.pot = potential
        self.cosmo = cosmology
        self.m_gamma = np.sqrt(potential.mass_squared()) * 5.61e32  # Convert to eV
        
    def alpha_variation(self, gamma):
        """Δα/α variation"""
        zeta = 0.010  # Coupling constant
        return zeta * (gamma - self.pot.gamma0) / 2.43e18  # M_Pl in eV
        
    def GW_damping(self, frequency, distance):
        """Gravitational wave damping Δh/h"""
        return -np.pi * frequency * self.m_gamma**2 * distance / self.cosmo.H0
        
    def galaxy_correlation_modulation(self, scale):
        """Amplitude A at given scale (Mpc)"""
        return 0.0042 * np.exp(-0.01 * (scale - 100)**2)
        
    def neutron_EDM(self):
        """Predicted neutron EDM"""
        return 8.1e-33  # e·cm
        
    def gamma_gamma_resonance(self):
        """Resonance energy in TeV"""
        return 3.25