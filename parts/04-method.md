# Method

## Overview

A machine learning model will be trained to use information from sinogram space to adjust images in image space. The network will be trained to perform this conversion in iterative steps similar to the iterative reconstruction method described in [@mechlem2018joint].

:::{figure} Figures/model2.png
:label: model2
Schematic overview of the model workflow. The input to the model consists of the first reconstructed image from the iterative algorithm (called iteration 0 for the AttU-Net model) combined with the corresponding projection (sinogram) data. The network processes this input iteratively, where the output of the $n^{th}$ iteration is used as the input for the $(n+1)^{th}$ iteration, while the projection data remains constant. After N iterations, the final reconstructed image is produced.
:::

An overview of the model workflow is shown in [](#model2). The network is designed to use the first reconstructed image from the iterative algorithm (called iteration 0 for the AttU-Net model) combined with the projection matrix and refine the reconstructed image across 20 iterations. The output image of an iteration is used as an input for the following iteration, making sure the projection matrix remains constant. After 20 iterations, the output of the network is used as the final image. 

## Generating phantoms 
The training data consists of phantoms generated according to a set of rules for which projections are simulated. These projections are used by the iterative algorithm, where the intermediate images, as illustrated by the red, dotted line in [](#itt), are used as training data for the model. The phantoms are randomly generated according to a set of rules: 

- each phantom is 32x32x32 pixels and consists of 2 channels, one for bone and one for water.
- each phantom has a main, elliptical body centered in the image consisting of a low mass density of both water and bone (both densities uniformly distributed between 0.1 and 0.6).
- The center of the body can be randomly offset by −3 to +3 pixels in both the x and y directions (uniform distribution). 
- The size of the main body varies, with the distance between the points on the axis varying between 1 and 3 pixels uniformly distributed. 
- Each phantom has 2 internal features, represented as smaller elliptical regions within the main body.
- Each feature has either a higher bone density or a higher water density than the other.
- The bone and/or water density cannot be lower in a feature than the main body

Each feature represents a distinct elliptical region. Because photon counting detectors (PCDs) allow differentiating tissue types, we assign one feature a higher bone density and the other a higher water density. This design ensures that the network is trained on a task relevant to the clinical use of PCDs. These rules have been implemented in python to generate a sample of phantoms as seen in the supporting [notebooks](#generate_phantom).

## Acquiring data

For 506 phantoms, detector measurements are simulated. These measurements are acquired with 32 angles per phantom. Next, each set of detector measurements are reconstructed using the iterative algorithm, the predefined endpoint as shown in [](#itt) is defined as 40 iterations. These iterations, illustrated by the red, dotted line in the same figure, are used to construct input-output pairs for training the model. Using a stride (how many iterations to skip between input and output images), the input is defined as the image from the $n^{th}$ iteration, with a corresponding output image taken from the $(n+stride)^{th}$ iteration.

Using a stride between input and output images allows the model to learn from a wider variety of scenarios within the same training time. In this work, a stride of 4 is used. This work made use of 506 phantoms, which generated 4554 input-output pairs.


## Attention based U-Net

To enable the network to learn how features in the image space and the projection (sinogram) space attend to eachother, a U-Net architecture with attention mechanisms integrated into the skip connections is used (AttU-Net) as described by [@oktay2018attentionunet]. The attention gates help the model to focus on which parts of the sinogram contribute to specific image regions.

The AttU-Net model is illustrated in [](#attunet). The architecture consists of five convolutional encoder blocks, a bottleneck block, and five decoder blocks. The number of channels increases through the encoder (64, 128, 256, 512 and 1024), and then decreases symmetrically through the decoder. The output layer applies a sigmoid activation to produce voxel-wise probabilities. The model has been implemented in Python using Pytorch in the supporting [notebooks](#network_code)

The model is trained on an NVDIA Tesla A100, with 4 hours allocated per run. In total 6 runs were completed, which resulted in 180 epochs (20 epochs per run). The model uses MSELoss as a loss function and Adam optimisation, which is a stochastic gradient descent method. The learning rate is set to 0.001, which is commonly used with Adam optimisation [@kingma2014adam]. An implementation of the training function can be seen in the supporting [notebooks](#train_network)

## Shaping the data

The AttU-Net model takes an image as input and produces an image of the same size as an output. Since 2 different images are used (the image and the projection), these must be combined into a single matrix. The projection matrix has dimensions (number of projections) x (z pixels detector) x (y pixels detector), where this work uses 32 projections and a detector of 64 by 44 pixels as set for the simulations in the supporting [notebooks](#settings). 

Aditionally, due to the pooling layers all dimensions must be powers of 16 as to not get an uneven amount of devisions. The projection matrix is originally 32 x 44 x 64 pixels, which we can pad with zeros to 32 x 48 x 64 pixels. The image will is 32 x 32 x 32 pixels which we can pad to 32 x 48 x 32 to fit the dimensions of the projections. The final input set to the network will be 10 x 32 x 48 x 96 as we use 10 channels in total. The output will be 2 x 32 x 48 x 96, 2 output channels corresponding the bone and water images.  

## Validation

20 phantoms are used as a validation set, generated under the same conditions and rules as those used for training. The performance of both the AttU-Net model and the iterative algorithm will be compared by calculating the root-mean-square (RMS) error between a reconstructed image and the ground truth (GT) image. The RMS error is calculated as

```{math}
\text{RMS\_error} = \sqrt{ \frac{1}{N} \sum_{i=1}^{N} \left( I_1(i) - I_2(i) \right)^2 }
```

where $I_1(i)$ and $I_2(i)$ are the pixel values at position $i$ in image 1 and image 2, respectively. N is the total number of pixels in the image, 32768 pixels in this case. 


