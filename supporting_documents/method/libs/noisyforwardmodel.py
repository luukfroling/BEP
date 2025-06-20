import numpy as np

def noisyforwardmodel(phantom, A, S, M):
    """
    Forward Model with Poisson Noise added.

    Parameters:
        phantom (ndarray): Phantom matrix of shape ((objectSize^3) x nMats)
        A (ndarray): Forward projection matrix of shape (nPixels x (objectSize^3))
        S (ndarray): Binned photon counts for each input energy (nBins x nEnergies)
        M (ndarray): Material attenuations for each input energy (nEnergies x nMats)

    Returns:
        noisyCounts (ndarray): Photon counts in each bin, for every ray and projection with Poisson noise (nBins x nPixels)
    """

    # Calculate the material integrals: 
    # This represents the total path length that a ray travels through each material
    # Shape: (nPixels x nMats)
    material_integrals = A @ phantom  # Matrix multiplication

    # Compute attenuation factor: How much each ray is attenuated based on material properties
    # Shape: (nEnergies x nPixels) = (nEnergies x nMats) * (nMats x nPixels)
    attenuation_factor = np.exp(-M @ material_integrals.T)

    # Compute the photon counts in each energy bin for every ray
    # Shape: (nBins x nPixels) = (nBins x nEnergies) * (nEnergies x nPixels)
    counts = S @ attenuation_factor

    # Apply Poisson noise to simulate photon noise
    noisy_counts = np.random.poisson(counts)        
    #noisy_counts = counts

    return noisy_counts
