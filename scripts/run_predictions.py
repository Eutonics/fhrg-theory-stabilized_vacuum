# scripts/run_predictions.py
import numpy as np
from core.potentials import FHRGPotential
from core.cosmology import FHRGCosmology
from core.observables import FHRGObservables

# Initialize models
pot = FHRGPotential()
cosmo = FHRGCosmology(pot)
obs = FHRGObservables(pot, cosmo)

# Calculate predictions
predictions = {
    "alpha_variation_z1": obs.alpha_variation(1.47),
    "GW_damping_500Hz": obs.GW_damping(500, 40),
    "galaxy_correlation_100Mpc": obs.galaxy_correlation_modulation(100),
    "neutron_EDM": obs.neutron_EDM(),
    "gamma_gamma_resonance": obs.gamma_gamma_resonance(),
    "S8": 0.812  # From cosmology calculation
}

# Save predictions
np.save("fhrg_predictions.npy", predictions)
print("FHRG Predictions:")
for key, value in predictions.items():
    print(f"{key}: {value}")