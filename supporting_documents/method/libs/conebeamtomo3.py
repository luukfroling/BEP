import numpy as np
import scipy.sparse as sp

def conebeamtomo3(objectSize, projAngles, pixelPitch, nPixelsY, nPixelsZ):
    """
    Generate the forward projection matrix for cone beam tomography.
    
    Parameters:
        objectSize (int): The length of one side of a cubed object.
        projAngles (ndarray): Array of projection angles (degrees).
        pixelPitch (float): Distance between detector pixels.
        nPixelsY (int): Number of pixels in the Y direction.
        nPixelsZ (int): Number of pixels in the Z direction.
    
    Returns:
        A (dict): Contains rows, cols, and vals for sparse matrix construction.
    """
    nAngles = len(projAngles)
    # print('nAngles:', nAngles)
    d_detector = 900
    dSource = 2300
    
    # Determining the number of pixels on the detector plate
    angle = np.degrees(np.arcsin(objectSize / np.sqrt(2) / dSource))
    nPixelsY_advised = 2 * np.ceil(np.tan(np.radians(angle)) * (d_detector + dSource) / pixelPitch)
    nPixelsZ_advised = (np.floor(((dSource + d_detector) * objectSize / 2 / (dSource - 1 / np.sqrt(2)) - pixelPitch / 2) / pixelPitch) + 1) * 2
    
    xObject = np.arange(-objectSize / 2, objectSize / 2 + 1)
    yObject = xObject
    zObject = xObject
    
    # Preallocate matrices
    rows = np.zeros(nPixelsY * nPixelsZ * nAngles * (2 * objectSize), dtype=int)
    cols = np.zeros_like(rows)
    vals = np.zeros(nPixelsY * nPixelsZ * nAngles * (2 * objectSize), dtype=float)
    idx_end = 0
    
    # Starting coordinates of the detector pixels
    x_start_detector = np.ones(nPixelsY * nPixelsZ) * d_detector
    y_start_detector = np.tile(np.arange(-(pixelPitch/2 + (nPixelsY/2 - 1) * pixelPitch), (pixelPitch/2 + (nPixelsY/2 - 1) * pixelPitch) + pixelPitch, pixelPitch), nPixelsZ)
    z_start_detector = np.repeat(np.arange(-(pixelPitch/2 + (nPixelsZ/2 - 1) * pixelPitch), (pixelPitch/2 + (nPixelsZ/2 - 1) * pixelPitch) + pixelPitch, pixelPitch), nPixelsY)

    # Fixed angles of the detector pixels with the source
    thetaY = np.array(np.degrees(np.arctan(y_start_detector / (dSource + d_detector))))
    thetaZ = np.array(np.degrees(np.arctan(z_start_detector / (dSource + d_detector))))
    
    # Starting coordinate for the source
    y_start_source = 0
    z_start_source = 0
    x_start_source = -dSource
        
    # Storing values from the for loop
    for i in range(1,nAngles+1):
        
        angle = projAngles[i-1]
        
        xDetector = np.array(x_start_detector * np.cos(np.radians(angle)) - y_start_detector * np.sin(np.radians(angle)))
        yDetector = np.array(x_start_detector * np.sin(np.radians(angle)) + y_start_detector * np.cos(np.radians(angle)))
        zDetector = np.array(z_start_detector)

        xSource = np.array(x_start_source * np.cos(np.radians(angle)) - y_start_source * np.sin(np.radians(angle)))
        ySource = np.array(x_start_source * np.sin(np.radians(angle)) + y_start_source * np.cos(np.radians(angle)))
        zSource = np.array(z_start_source)
        
        if projAngles[i-1] <= np.degrees(np.arcsin(objectSize / (2 * dSource))):
            # Find coordinates where the beam enters the object
            # This is for all the pixels at the same time
            # x coordinate is -objectSize/2
            # Determine the y and z coordinates for the given x coordinate
            yCoordinateIn = np.array(ySource + (yDetector - ySource) * (-objectSize / 2 - xSource) / (xDetector - xSource))
            zCoordinateIn = np.array(zSource + (zDetector - zSource) * (-objectSize / 2 - xSource) / (xDetector - xSource))
            
            # Make a map where the values lie outside the object and thus the rays are to be discarded
            map_invalid = (yCoordinateIn < -objectSize / 2) | (yCoordinateIn > objectSize / 2) | \
                          (zCoordinateIn < -objectSize / 2) | (zCoordinateIn > objectSize / 2)
            valid_indices = ~map_invalid
            yCoordinateIn = yCoordinateIn[valid_indices]
            zCoordinateIn = zCoordinateIn[valid_indices]
            
            # Generate the right number of x coordinates
            xCoordinateIn = np.full(yCoordinateIn.shape[0], -objectSize / 2)
            
            index = np.arange(0,xDetector.shape[0])
            #print('xDetector:',xDetector.shape)
            #print('index:', index)
            index = index[valid_indices]
            #print(f"i: {i}, valid_indices: {valid_indices}")
            
            # Find the coordinates for the place the ray exits the object
            yCoordinateOut = np.array(np.array(ySource + (yDetector - ySource) * (objectSize / 2 - xSource) / (xDetector - xSource)))
            zCoordinateOut = np.array(zSource + (zDetector - zSource) * (objectSize / 2 - xSource) / (xDetector - xSource))
        
            yCoordinateOut = yCoordinateOut[valid_indices]
            zCoordinateOut = zCoordinateOut[valid_indices]
            
            xCoordinateOut = np.full(yCoordinateOut.shape[0], objectSize / 2)
        
            # Identify rays exiting through different sides of the object
            map_lower = yCoordinateOut < -objectSize / 2
            map_higher = yCoordinateOut > objectSize / 2
        
            # Prepare detector arrays for easier use
            xD = xDetector[valid_indices]
            yD = yDetector[valid_indices]
            zD = zDetector[valid_indices]
        
            # The angles
            anglesY = thetaY[valid_indices]
            anglesZ = thetaZ[valid_indices]
            
            # Adjust coordinates for rays exiting through the y-boundaries
            yCoordinateOut[map_higher] = objectSize / 2
            xCoordinateOut[map_higher] = xSource + (xD[map_higher] - xSource) * (objectSize / 2 - ySource) / (yD[map_higher] - ySource)
            zCoordinateOut[map_higher] = zSource + (zD[map_higher] - zSource) * (objectSize / 2 - ySource) / (yD[map_higher] - ySource)
        
            yCoordinateOut[map_lower] = -objectSize / 2
            xCoordinateOut[map_lower] = xSource + (xD[map_lower] - xSource) * (-objectSize / 2 - ySource) / (yD[map_lower] - ySource)
            zCoordinateOut[map_lower] = zSource + (zD[map_lower] - zSource) * (-objectSize / 2 - ySource) / (yD[map_lower] - ySource)
        
            # Identify rays exiting through the z-boundaries
            map_lower = zCoordinateOut < -objectSize / 2
            map_higher = zCoordinateOut > objectSize / 2
        
            # Adjust coordinates for rays exiting through the z-boundaries
            zCoordinateOut[map_higher] = objectSize / 2
            xCoordinateOut[map_higher] = xSource + (xD[map_higher] - xSource) * (objectSize / 2 - zSource) / (zD[map_higher] - zSource)
            yCoordinateOut[map_higher] = ySource + (yD[map_higher] - ySource) * (objectSize / 2 - zSource) / (zD[map_higher] - zSource)
        
            zCoordinateOut[map_lower] = -objectSize / 2
            xCoordinateOut[map_lower] = xSource + (xD[map_lower] - xSource) * (-objectSize / 2 - zSource) / (zD[map_lower] - zSource)
            yCoordinateOut[map_lower] = ySource + (yD[map_lower] - ySource) * (-objectSize / 2 - zSource) / (zD[map_lower] - zSource)

        elif projAngles[i-1] <= (90 - np.degrees(np.arcsin(objectSize / 2 / dSource))):
            # Find coordinates where the beam enters the object
            # This is for all the pixels at the same time
            # x coordinate is -objectSize/2
            # Determine the y and z coordinates for the given x coordinate
            yCoordinateIn = ySource + (yDetector - ySource) * (-objectSize / 2 - xSource) / (xDetector - xSource)
            zCoordinateIn = zSource + (zDetector - zSource) * (-objectSize / 2 - xSource) / (xDetector - xSource)
            
            # Make a map where the values lie outside the object and thus the rays are to be discarded
            map_invalid = (yCoordinateIn > objectSize / 2) | (zCoordinateIn < -objectSize / 2) | (zCoordinateIn > objectSize / 2)
            valid_indices = ~map_invalid
            yCoordinateIn = yCoordinateIn[valid_indices]
            zCoordinateIn = zCoordinateIn[valid_indices]
        
            # Generate the right number of x coordinates
            xCoordinateIn = np.full(yCoordinateIn.shape[0], -objectSize / 2)
        
            index = np.arange(0,xDetector.shape[0])
            index = index[~map_invalid]
        
            # Preparing the detector arrays so that they are easier to use in the coming code
            xD = xDetector[valid_indices]
            yD = yDetector[valid_indices]
            zD = zDetector[valid_indices]
        
            # Check the coordinates for going in on the y-side
            mapY = yCoordinateIn < -objectSize / 2
            yCoordinateIn[mapY] = -objectSize / 2
            xCoordinateIn[mapY] = xSource + (xD[mapY] - xSource) * (-objectSize / 2 - ySource) / (yD[mapY] - ySource)
            zCoordinateIn[mapY] = zSource + (zD[mapY] - zSource) * (-objectSize / 2 - ySource) / (yD[mapY] - ySource)
        
            # Check if there are still some rays outside of the object
            mapX = xCoordinateIn > objectSize / 2
            validX_indices = ~mapX
            xCoordinateIn = xCoordinateIn[validX_indices]
            yCoordinateIn = yCoordinateIn[validX_indices]
            zCoordinateIn = zCoordinateIn[validX_indices]
            index = index[validX_indices]
        
            # Make a smaller array for the angles (discard all unnecessary angles)
            anglesY = thetaY[valid_indices]
            anglesZ = thetaZ[valid_indices]
        
            anglesZ = anglesZ[validX_indices]
            anglesY = anglesY[validX_indices]
        
            xD = xD[validX_indices]
            yD = yD[validX_indices]
            zD = zD[validX_indices]
        
            # Find the coordinates for the place the ray exits the object
            yCoordinateOut = np.array(ySource + (yD - ySource) * (objectSize / 2 - xSource) / (xD - xSource))
            zCoordinateOut = np.array(zSource + (zD - zSource) * (objectSize / 2 - xSource) / (xD - xSource))
        
            # The fixed x coordinates
            xCoordinateOut = np.full(yCoordinateOut.shape[0], objectSize / 2)
        
            # Identify rays exiting through different sides of the object
            map_higher = yCoordinateOut > objectSize / 2
        
            # Changing the coordinates in the higher case
            yCoordinateOut[map_higher] = objectSize / 2
            xCoordinateOut[map_higher] = xSource + (xD[map_higher] - xSource) * (objectSize / 2 - ySource) / (yD[map_higher] - ySource)
            zCoordinateOut[map_higher] = zSource + (zD[map_higher] - zSource) * (objectSize / 2 - ySource) / (yD[map_higher] - ySource)
        
            # What if the ray goes out of the phantom through the z-edges
            map_lower = zCoordinateOut < -objectSize / 2
            map_higher = zCoordinateOut > objectSize / 2
        
            # Changing the coordinates in the higher case
            zCoordinateOut[map_higher] = objectSize / 2
            xCoordinateOut[map_higher] = xSource + (xD[map_higher] - xSource) * (objectSize / 2 - zSource) / (zD[map_higher] - zSource)
            yCoordinateOut[map_higher] = ySource + (yD[map_higher] - ySource) * (objectSize / 2 - zSource) / (zD[map_higher] - zSource)
        
            # Changing the coordinates in the lower case
            zCoordinateOut[map_lower] = -objectSize / 2
            xCoordinateOut[map_lower] = xSource + (xD[map_lower] - xSource) * (-objectSize / 2 - zSource) / (zD[map_lower] - zSource)
            yCoordinateOut[map_lower] = ySource + (yD[map_lower] - ySource) * (-objectSize / 2 - zSource) / (zD[map_lower] - zSource)
            
        elif projAngles[i-1] <= (90 + np.degrees(np.arcsin(objectSize / 2 / dSource))):
            # Find coordinates where the beam enters the object
            # This is for all the pixels at the same time
            # y coordinate is -objectSize/2
            # Determine the x and z coordinates for the given y coordinate
            xCoordinateIn = xSource + (xDetector - xSource) * (-objectSize / 2 - ySource) / (yDetector - ySource)
            zCoordinateIn = zSource + (zDetector - zSource) * (-objectSize / 2 - ySource) / (yDetector - ySource)
        
            # Make a map where the values lie outside the object and thus the rays are to be discarded
            map_invalid = (xCoordinateIn < -objectSize / 2) | (xCoordinateIn > objectSize / 2) | \
                          (zCoordinateIn < -objectSize / 2) | (zCoordinateIn > objectSize / 2)
            valid_indices = ~map_invalid
            xCoordinateIn = xCoordinateIn[valid_indices]
            zCoordinateIn = zCoordinateIn[valid_indices]
        
            # Generate the right number of y coordinates
            yCoordinateIn = np.full(xCoordinateIn.shape[0], -objectSize / 2)
        
            index = np.arange(0,xDetector.shape[0])
            index = index[valid_indices]
        
            # Preparing the detector arrays so that they are easier to use in the coming code
            xD = xDetector[valid_indices]
            yD = yDetector[valid_indices]
            zD = zDetector[valid_indices]
        
            anglesY = thetaY[valid_indices]
            anglesZ = thetaZ[valid_indices]
        
            # Find the coordinates for the place the ray exits the object
            xCoordinateOut = xSource + (xD - xSource) * (objectSize / 2 - ySource) / (yD - ySource)
            zCoordinateOut = zSource + (zD - zSource) * (objectSize / 2 - ySource) / (yD - ySource)
        
            # The fixed y coordinates
            yCoordinateOut = np.full(xCoordinateOut.shape[0], objectSize / 2)
        
            # Identify rays exiting through different sides of the object
            map_lower = xCoordinateOut < -objectSize / 2
            map_higher = xCoordinateOut > objectSize / 2
        
            # Changing the coordinates in the higher case
            xCoordinateOut[map_higher] = objectSize / 2
            yCoordinateOut[map_higher] = ySource + (yD[map_higher] - ySource) * (objectSize / 2 - xSource) / (xD[map_higher] - xSource)
            zCoordinateOut[map_higher] = zSource + (zD[map_higher] - zSource) * (objectSize / 2 - xSource) / (xD[map_higher] - xSource)
        
            # Changing the coordinates in the lower case
            xCoordinateOut[map_lower] = -objectSize / 2
            yCoordinateOut[map_lower] = ySource + (yD[map_lower] - ySource) * (-objectSize / 2 - xSource) / (xD[map_lower] - xSource)
            zCoordinateOut[map_lower] = zSource + (zD[map_lower] - zSource) * (-objectSize / 2 - xSource) / (xD[map_lower] - xSource)
        
            # Identify rays exiting through the z-boundaries
            map_lower = zCoordinateOut < -objectSize / 2
            map_higher = zCoordinateOut > objectSize / 2
        
            # Changing the coordinates in the higher case
            zCoordinateOut[map_higher] = objectSize / 2
            xCoordinateOut[map_higher] = xSource + (xD[map_higher] - xSource) * (objectSize / 2 - zSource) / (zD[map_higher] - zSource)
            yCoordinateOut[map_higher] = ySource + (yD[map_higher] - ySource) * (objectSize / 2 - zSource) / (zD[map_higher] - zSource)
        
            # Changing the coordinates in the lower case
            zCoordinateOut[map_lower] = -objectSize / 2
            xCoordinateOut[map_lower] = xSource + (xD[map_lower] - xSource) * (-objectSize / 2 - zSource) / (zD[map_lower] - zSource)
            yCoordinateOut[map_lower] = ySource + (yD[map_lower] - ySource) * (-objectSize / 2 - zSource) / (zD[map_lower] - zSource)

        elif projAngles[i-1] <= (180 - np.degrees(np.arcsin(objectSize / 2 / dSource))):
            # Find coordinates where the beam enters the object
            # This is for all the pixels at the same time
            # x coordinate is objectSize/2
            # Determine the y and z coordinates for the given x coordinate
            yCoordinateIn = ySource + (yDetector - ySource) * (objectSize / 2 - xSource) / (xDetector - xSource)
            zCoordinateIn = zSource + (zDetector - zSource) * (objectSize / 2 - xSource) / (xDetector - xSource)
        
            # Make a map where the values lie outside the object and thus the rays are to be discarded
            map_invalid = (yCoordinateIn > objectSize / 2) | (zCoordinateIn < -objectSize / 2) | (zCoordinateIn > objectSize / 2)
            valid_indices = ~map_invalid
            yCoordinateIn = yCoordinateIn[valid_indices]
            zCoordinateIn = zCoordinateIn[valid_indices]
        
            # Generate the right number of x coordinates
            xCoordinateIn = np.full(yCoordinateIn.shape[0], objectSize / 2)
        
            index = np.arange(0,xDetector.shape[0])
            index = index[valid_indices]
        
            # Preparing the detector arrays so that they are easier to use in the coming code
            xD = xDetector[valid_indices]
            yD = yDetector[valid_indices]
            zD = zDetector[valid_indices]
        
            # Check the coordinates for going in on the y-side
            mapY = yCoordinateIn < -objectSize / 2
            yCoordinateIn[mapY] = -objectSize / 2
            xCoordinateIn[mapY] = xSource + (xD[mapY] - xSource) * (-objectSize / 2 - ySource) / (yD[mapY] - ySource)
            zCoordinateIn[mapY] = zSource + (zD[mapY] - zSource) * (-objectSize / 2 - ySource) / (yD[mapY] - ySource)
        
            # Check if there are still some rays outside of the object
            mapX = xCoordinateIn < -objectSize / 2
            validX_indices = ~mapX
            xCoordinateIn = xCoordinateIn[validX_indices]
            yCoordinateIn = yCoordinateIn[validX_indices]
            zCoordinateIn = zCoordinateIn[validX_indices]
            index = index[validX_indices]
        
            anglesY = thetaY[valid_indices]
            anglesZ = thetaZ[valid_indices]
            
            anglesZ = anglesZ[validX_indices]
            anglesY = anglesY[validX_indices]
        
            xD = xD[validX_indices]
            yD = yD[validX_indices]
            zD = zD[validX_indices]
        
            # Find the coordinates for the place the ray exits the object
            yCoordinateOut = ySource + (yD - ySource) * (-objectSize / 2 - xSource) / (xD - xSource)
            zCoordinateOut = zSource + (zD - zSource) * (-objectSize / 2 - xSource) / (xD - xSource)
        
            # The fixed x coordinates
            xCoordinateOut = np.full(yCoordinateOut.shape[0], -objectSize / 2)
        
            # Identify rays exiting through different sides of the object
            map_higher = yCoordinateOut > objectSize / 2
        
            # Changing the coordinates in the higher case
            yCoordinateOut[map_higher] = objectSize / 2
            xCoordinateOut[map_higher] = xSource + (xD[map_higher] - xSource) * (objectSize / 2 - ySource) / (yD[map_higher] - ySource)
            zCoordinateOut[map_higher] = zSource + (zD[map_higher] - zSource) * (objectSize / 2 - ySource) / (yD[map_higher] - ySource)
        
            # Identify rays exiting through the z-boundaries
            map_lower = zCoordinateOut < -objectSize / 2
            map_higher = zCoordinateOut > objectSize / 2
        
            # Changing the coordinates in the higher case
            zCoordinateOut[map_higher] = objectSize / 2
            xCoordinateOut[map_higher] = xSource + (xD[map_higher] - xSource) * (objectSize / 2 - zSource) / (zD[map_higher] - zSource)
            yCoordinateOut[map_higher] = ySource + (yD[map_higher] - ySource) * (objectSize / 2 - zSource) / (zD[map_higher] - zSource)
        
            # Changing the coordinates in the lower case
            zCoordinateOut[map_lower] = -objectSize / 2
            xCoordinateOut[map_lower] = xSource + (xD[map_lower] - xSource) * (-objectSize / 2 - zSource) / (zD[map_lower] - zSource)
            yCoordinateOut[map_lower] = ySource + (yD[map_lower] - ySource) * (-objectSize / 2 - zSource) / (zD[map_lower] - zSource)
        else:
            # Find coordinates where the beam enters the object
            # This is for all the pixels at the same time
            # x coordinate is objectSize/2
            # Determine the y and z coordinates for the given x coordinate
            yCoordinateIn = ySource + (yDetector - ySource) * (objectSize / 2 - xSource) / (xDetector - xSource)
            zCoordinateIn = zSource + (zDetector - zSource) * (objectSize / 2 - xSource) / (xDetector - xSource)
        
            # Make a map where the values lie outside the object and thus the rays are to be discarded
            map_invalid = (yCoordinateIn < -objectSize / 2) | (yCoordinateIn > objectSize / 2) | \
                          (zCoordinateIn < -objectSize / 2) | (zCoordinateIn > objectSize / 2)
            valid_indices = ~map_invalid
            yCoordinateIn = yCoordinateIn[valid_indices]
            zCoordinateIn = zCoordinateIn[valid_indices]
        
            # Generate the right number of x coordinates
            xCoordinateIn = np.full(yCoordinateIn.shape[0], objectSize / 2)
        
            index = np.arange(0,xDetector.shape[0])
            index = index[valid_indices]
        
            # Find the coordinates for the place the ray exits the object
            yCoordinateOut = ySource + (yDetector - ySource) * (-objectSize / 2 - xSource) / (xDetector - xSource)
            zCoordinateOut = zSource + (zDetector - zSource) * (-objectSize / 2 - xSource) / (xDetector - xSource)
        
            yCoordinateOut = yCoordinateOut[valid_indices]
            zCoordinateOut = zCoordinateOut[valid_indices]
        
            # The fixed x coordinates
            xCoordinateOut = np.full(yCoordinateOut.shape[0], -objectSize / 2)
        
            # Identify rays exiting through different sides of the object
            map_lower = yCoordinateOut < -objectSize / 2
            map_higher = yCoordinateOut > objectSize / 2
        
            # Preparing the detector arrays so that they are easier to use in the coming code
            xD = xDetector[valid_indices]
            yD = yDetector[valid_indices]
            zD = zDetector[valid_indices]
        
            anglesY = thetaY[valid_indices]
            anglesZ = thetaZ[valid_indices]
        
            # Changing the coordinates in the higher case
            yCoordinateOut[map_higher] = objectSize / 2
            xCoordinateOut[map_higher] = xSource + (xD[map_higher] - xSource) * (objectSize / 2 - ySource) / (yD[map_higher] - ySource)
            zCoordinateOut[map_higher] = zSource + (zD[map_higher] - zSource) * (objectSize / 2 - ySource) / (yD[map_higher] - ySource)
        
            # Changing the coordinates in the lower case
            yCoordinateOut[map_lower] = -objectSize / 2
            xCoordinateOut[map_lower] = xSource + (xD[map_lower] - xSource) * (-objectSize / 2 - ySource) / (yD[map_lower] - ySource)
            zCoordinateOut[map_lower] = zSource + (zD[map_lower] - zSource) * (-objectSize / 2 - ySource) / (yD[map_lower] - ySource)
        
            # Identify rays exiting through the z-boundaries
            map_lower = zCoordinateOut < -objectSize / 2
            map_higher = zCoordinateOut > objectSize / 2
        
            # Changing the coordinates in the higher case
            zCoordinateOut[map_higher] = objectSize / 2
            xCoordinateOut[map_higher] = xSource + (xD[map_higher] - xSource) * (objectSize / 2 - zSource) / (zD[map_higher] - zSource)
            yCoordinateOut[map_higher] = ySource + (yD[map_higher] - ySource) * (objectSize / 2 - zSource) / (zD[map_higher] - zSource)
        
            # Changing the coordinates in the lower case
            zCoordinateOut[map_lower] = -objectSize / 2
            xCoordinateOut[map_lower] = xSource + (xD[map_lower] - xSource) * (-objectSize / 2 - zSource) / (zD[map_lower] - zSource)
            yCoordinateOut[map_lower] = ySource + (yD[map_lower] - ySource) * (-objectSize / 2 - zSource) / (zD[map_lower] - zSource)
            
        # Now to deal with finding the other intersections inside the object
        # print('xCoordinateIn:', xCoordinateIn.shape)
        # print('--------------')
        for n in range(1,xCoordinateIn.shape[0]+1):
            # First the intersection points for the x coordinates
            if (projAngles[i-1] + anglesY[n-1]) < 90:
                if (np.all(np.floor(xCoordinateOut[n-1]) == np.floor(xCoordinateIn[n-1])) and (np.ceil(xCoordinateOut[n-1]) == np.ceil(xCoordinateIn[n-1]))):
                    x = np.array([])
                else:
                    x = np.arange(np.ceil(xCoordinateIn[n-1]), np.floor(xCoordinateOut[n-1])+1)
            else:
                if (np.all(np.floor(xCoordinateOut[n-1]) == np.floor(xCoordinateIn[n-1])) and (np.ceil(xCoordinateOut[n-1]) == np.ceil(xCoordinateIn[n-1]))):
                    x = np.array([])
                else:
                    x = np.arange(np.ceil(xCoordinateOut[n-1]), np.floor(xCoordinateIn[n-1])+1)
        
            # Intersection points for the y coordinates
            if (((projAngles[i-1] + anglesY[n-1]) > 180) or ((projAngles[i-1] + anglesY[n-1]) < 0)):
                if (np.all(np.floor(yCoordinateOut[n-1]) == np.floor(yCoordinateIn[n-1])) and (np.ceil(yCoordinateOut[n-1]) == np.ceil(yCoordinateIn[n-1]))):
                    y = np.array([])
                else:
                    y = np.arange(np.ceil(yCoordinateOut[n-1]), np.floor(yCoordinateIn[n-1])+1)
            else:
                if (np.all(np.floor(yCoordinateOut[n-1]) == np.floor(yCoordinateIn[n-1])) and (np.ceil(yCoordinateOut[n-1]) == np.ceil(yCoordinateIn[n-1]))):
                    y = np.array([])
                else:
                    y = np.arange(np.ceil(yCoordinateIn[n-1]), np.floor(yCoordinateOut[n-1])+1)
        
            # Intersection points for the z coordinates
            if projAngles[i-1] == 92 and n == 0:
                pass
            
            if anglesZ[n-1] > 0:
                if (np.all(np.floor(zCoordinateOut[n-1]) == np.floor(zCoordinateIn[n-1])) and (np.ceil(zCoordinateOut[n-1]) == np.ceil(zCoordinateIn[n-1]))):
                    z = np.array([])
                else:
                    z = np.arange(np.ceil(zCoordinateIn[n-1]), np.floor(zCoordinateOut[n-1])+1)
            else:
                if (np.all(np.floor(zCoordinateOut[n-1]) == np.floor(zCoordinateIn[n-1])) and (np.ceil(zCoordinateOut[n-1]) == np.ceil(zCoordinateIn[n-1]))):
                    z = np.array([])
                else:
                    z = np.arange(np.ceil(zCoordinateOut[n-1]), np.floor(zCoordinateIn[n-1])+1)
                    
            # Slope Z
            xIntersectZ = xSource + (xD[n-1] - xSource) * (z - zSource) / (zD[n-1] - zSource)
            yIntersectZ = ySource + (yD[n-1] - ySource) * (z - zSource) / (zD[n-1] - zSource)
        
            # Slope X
            yIntersectX = ySource + (yD[n-1] - ySource) * (x - xSource) / (xD[n-1] - xSource)
            zIntersectX = zSource + (zD[n-1] - zSource) * (x - xSource) / (xD[n-1] - xSource)
        
            # Slope Y
            xIntersectY = xSource + (xD[n-1] - xSource) * (y - ySource) / (yD[n-1] - ySource)
            zIntersectY = zSource + (zD[n-1] - zSource) * (y - ySource) / (yD[n-1] - ySource)
        
            # Combine the intersection points as coordinates of all intersections with either x, y, or z.
            xcoor = np.hstack([x, xIntersectY, xIntersectZ])
            ycoor = np.hstack([yIntersectX, y, yIntersectZ])
            zcoor = np.hstack([zIntersectX, zIntersectY, z])
            
            if xSource <= 0:
                map3 = np.argsort(xcoor)
            else:
                map3 = np.argsort(xcoor)[::-1]
               
            xcoor = xcoor[map3]
            ycoor = ycoor[map3]
            zcoor = zcoor[map3]

            # Calculate the length within cell and determines the number of cells which is hit.
            d = np.sqrt(np.diff(xcoor, axis=0) ** 2 + np.diff(ycoor, axis=0) ** 2 + np.diff(zcoor, axis=0) ** 2)
            numvals = d.size
        
            # Store the values inside the box.
            if numvals > 0:
                # Calculates the midpoints of the line within the cells.
                xMP = 0.5 * (xcoor[:-1] + xcoor[1:]) + objectSize / 2
                yMP = 0.5 * (ycoor[:-1] + ycoor[1:]) + objectSize / 2
                zMP = 0.5 * (zcoor[:-1] + zcoor[1:]) + objectSize / 2

                # Translate the midpoint coordinates to index.
                col = np.floor(xMP) * objectSize + (objectSize - np.floor(yMP)) + (objectSize ** 2) * (objectSize - np.ceil(zMP)) - 1
                
                # Create the indices to store the values to vector for later creation of A matrix.
                idxstart = idx_end + 1
                idx_end = idxstart + numvals - 1
                idx = np.arange(idxstart, idx_end + 1)

                # Store row numbers, column numbers, and values.
                rows[idx] = (i-1) * nPixelsY * nPixelsZ + index[n-1] 
                cols[idx] = col
                vals[idx] = d / 10
    
    # Truncate excess zeros.
    rows = rows[1:idx_end+1]
    cols = cols[1:idx_end+1]
    vals = vals[1:idx_end+1]
    
    map_invalid = cols < 0
    
    rows = rows.ravel()
    cols = cols.ravel()
    vals = vals.ravel()
    
    # Create sparse matrix A from the stored values.
    A = sp.csr_matrix((vals, (rows, cols)), shape=(nPixelsY * nPixelsZ * nAngles, objectSize ** 3 ))
    return A

                            