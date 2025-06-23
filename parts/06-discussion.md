# Discussion
## Model is only as good as your data

The AttU-Net model does recognise features present in the phantom but fails to iteratively get closer to the ground truth. This is apperent from the features present in the reconstructed images of the first iterations shown in [](#bone_recons), and the convergence to a higher RMS error than the iterative aproach for all phantoms shown in [](#reconPhantom). Part of this result can be explained using the common phrase 'a model is only as good as the data it gets'. 

Looking at the reconstruction results shown in [](#wrong), it is clear the iterative algorithm does not always generate reliable results. Similar images will have been used during training as well, which can negatively impact the training results. For future research it's recommended to only use images reconstructed using the iterative aproach with a rms error smaller than a set threshold to ensure quality of the data.

## non-machine learning speedup

Looking at the reconstruction time, the AttU-Net is more optimised than the iterative algorithm. The AttU-Net makes use of the PyTorch framework which is highly optimised for machine learning tasks. The iterative algorithm has been implemented from scratch and does not include any form of optimisations. For future research it is recommended to also optimise the iterative approach by, for example, performing the matrix multiplications from equation [](#projection_eq) in parallel. This makes for a more accurate comparison of the rms error against time for the two methods.

## Image size, model size, number of projections

Currently, it's hard to predict how the running time of the AttU-Net model will grow as the input image grows. The model currently takes 0.5 GB of storage for a 32 x 32 x 32 image. Testing the model for bigger images will require a larger number of projections as well to capture all the image. On top of that the number of channels within the network must increase for each layer to accomodate a larger feature map. To properly compare running times, one must a build model similar to the AtttU-Net model which reconstructs larger images with the same average error. This makes sure no accuracy is lost during comparison. 


## Adding cross attention between spaces

adding cross attention can more accuratly predict which part of the sinogram attend to which parts of the image. Now two different spaces, image space and projection space, are added into a single matrix. Cross attention might be able to seperate these better. 

## regularisation term

Looking at the image reconstructed by the AttU-Net model for the bone density seen in [](#wrong_reconstruction), the model could perform better if a regularisation term is added. Look into possibility of performing iterative algorithm after a certain speedup with the proposed model. 