import numpy as np
import scipy.io
from libs.conebeamtomo3 import conebeamtomo3
from libs.noisyforwardmodel import noisyforwardmodel

def projectMatrix(x, object_size, n_pixels_y, n_pixels_z, pixel_pitch, n_proj = 100):
    """
    Generate the forward projection matrix.

    Parameters:
        x (ndarray): Object data (nMats x (object_size^3)).
        object_size (int): Length of one side of a cubed object.
        n_pixels_y (int): Number of pixels in Y direction.
        n_pixels_z (int): Number of pixels in Z direction.
        pixel_pitch (float): Distance between the centers of the pixels in mm.

    Returns:
        A (ndarray): Forward Projection matrix (nPixels x (object_size^3)).
    """
    
    # Load MATLAB .mat files containing the required data
    incident_spectrum = scipy.io.loadmat('libs/incidentSpectrum.mat')['incidentSpectrum']
    material_attenuations = scipy.io.loadmat('libs/materialAttenuationBoneWater.mat')['materialAttenuations']
    detector_response = scipy.io.loadmat('libs/detectorResponse.mat')['detectorResponse']

    # Material attenuations (nEnergies x nMats)
    M = material_attenuations

    # sUnbinned: (180 x nEnergies) = detectorResponse .* incidentSpectrum'
    s_unbinned = detector_response * incident_spectrum.T

    # Define thresholds and binning (nBins x nEnergies)
    S = np.vstack([
        np.sum(s_unbinned[29:50, :], axis=0),
        np.sum(s_unbinned[50:61, :], axis=0),
        np.sum(s_unbinned[61:71, :], axis=0),
        np.sum(s_unbinned[71:82, :], axis=0),
        np.sum(s_unbinned[82:, :], axis=0),
    ])
    
    # np.savetxt("s_python.txt", S, fmt="%.16f", delimiter="\t")

    # Define projection angles
    projAngles = np.arange(0, n_proj)*(180/n_proj)  # Projection angles
    
    # Generate forward projection matrix
    A = conebeamtomo3(object_size, projAngles, pixel_pitch, n_pixels_y, n_pixels_z)

    y = noisyforwardmodel(x, A, S, M)

    return y, S, M, A
