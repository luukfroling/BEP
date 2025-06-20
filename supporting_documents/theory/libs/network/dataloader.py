import torch
from torch.utils.data import Dataset
import numpy as np

class CurriculumDataset(Dataset):
    def __init__(self, matrices, kind):  # kind = 'basic', 'random', or 'mixed'
        if kind == 'basic':
            self.bones, self.waters, _, self.ys = matrices
        else:
            self.bones = [mat['bone'] for mat in matrices]
            self.waters = [mat['water'] for mat in matrices]
            self.ys = [mat['y'] for mat in matrices]

    def __len__(self):
        return len(self.ys)

    def __getitem__(self, idx):
        input_matrix = prepareInputCurriculum(self.ys[idx])
        output_matrix = prepareOutputCurriculum(self.waters[idx], self.bones[idx])

        # Use from_numpy if input/output are NumPy arrays
        if isinstance(input_matrix, np.ndarray):
            input_tensor = torch.from_numpy(input_matrix).float()
        else:
            input_tensor = torch.tensor(input_matrix, dtype=torch.float32)

        if isinstance(output_matrix, np.ndarray):
            output_tensor = torch.from_numpy(output_matrix).float()
        else:
            output_tensor = torch.tensor(output_matrix, dtype=torch.float32)

        return input_tensor, output_tensor
    
class PhantomDataset(Dataset):
    def __init__(self, matrices):  # kind = 'basic', 'random', or 'mixed'
        # these correspond to the keys of the objects from the multiprocessing file.
        self.bones = [mat['bone'] for mat in matrices]
        self.waters = [mat['water'] for mat in matrices]
        self.ys = [mat['y'] for mat in matrices]

    def __len__(self):
        return len(self.ys)

    def __getitem__(self, idx):
        input_matrix = prepareInputCurriculum(self.ys[idx])
        output_matrix = prepareOutputCurriculum(self.waters[idx], self.bones[idx])

        input_tensor = torch.tensor(input_matrix, dtype=torch.float32)
        output_tensor = torch.tensor(output_matrix, dtype=torch.float32)

        return input_tensor, output_tensor


# ===== PREPARE DATA FUNCTIONS =====

def prepareInputCurriculum(y, objectSize = 32, projections = 32, nPixelsY= 64, nPixelsZ = 44):
    
    # Curriculum learning input preparation function
    # As an input: 10 channels consisting of empty images next to the sinograms

    # shape: 32x48x96, beginning with a 32x32x32 zero image with padding and a 32x44x64 image with padding

    zeroImage = np.zeros((objectSize, objectSize + 16, objectSize))
    zeroBin = np.zeros((projections, 2, nPixelsY))
    
    inputs = []

    # we have one dataset per y! with 10 channels (one for bone and one for water)
    for i, bins in enumerate(y):
        # no that is not correct
        
        channels = []
        # input is 10 channels, zeroimage appended to the energy bin
        bin = bins.reshape((projections, nPixelsZ, nPixelsY))
        
        # normalize so biggest value is 1, smalles value is 0
        bin = (bin - np.min(bin)) / (np.max(bin) - np.min(bin))
        
        bin = np.concatenate((zeroBin, bin, zeroBin), axis=1)
        # for input image, concatinate bin to zeroImage in x direction
        input_image = np.concatenate((zeroImage, bin), axis=2)
        inputs.append(input_image)
        inputs.append(input_image)
    
    return inputs

def prepareOutputCurriculum(water, bone, projections = 32, nPixelsY= 64, nPixelsZ = 44):

    water = np.flip(water, axis=1)  # flip along z-axis
    water = np.flip(water, axis=2)  # flip along y-axis
    bone = np.flip(bone, axis=1)    # flip along z-axis
    bone = np.flip(bone, axis=2)    # flip along y-axis
    
    zeroImage = np.zeros((32, 8, 32))
    zeroBin = np.zeros((projections, nPixelsZ + 4, nPixelsY))  # zero bin for the energy bin
    outputs = []
    
    bone_output = np.concatenate((zeroImage, bone, zeroImage), axis=1)
    water_output = np.concatenate((zeroImage, water, zeroImage), axis=1)

    bone_output = np.concatenate((bone_output, zeroBin), axis=2)
    water_output = np.concatenate((water_output, zeroBin), axis=2)
    
    outputs = [bone_output, water_output]  # shape will be (10, 32, 44, 64)
    
    return outputs
