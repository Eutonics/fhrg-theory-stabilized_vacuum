# scripts/run_cosmology.py
import numpy as np
import matplotlib.pyplot as plt
from core.potentials import FHRGPotential
from core.cosmology import FHRGCosmology

# Initialize potential and cosmology
pot = FHRGPotential()
cosmo = FHRGCosmology(pot)

# Solve cosmological equations
sol = cosmo.solve(z_span=[0, 5], initial_conditions=[1.45, 0.1])
z = sol.t
gamma, gamma_dot = sol.y

# Calculate observables
w = [cosmo.equation_of_state(zi, [g, d]) for zi, g, d in zip(z, gamma, gamma_dot)]
H = [cosmo.Hubble_parameter(zi, [g, d]) for zi, g, d in zip(z, gamma, gamma_dot)]
S8 = cosmo.S8_calculation(z, gamma)

# Save results
np.savez('cosmology_results.npz', z=z, gamma=gamma, w=w, H=H, S8=S8)

print(f"Predicted S₈ = {S8[-1]:.3f}")
print(f"w(z=0) = {w[0]:.4f}")
print(f"H₀ = {H[0]:.1f} km/s/Mpc")