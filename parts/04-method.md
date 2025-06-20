## Method

### Generating phantoms 

The phantoms, as illustrated in [](#phantomSinogram), are randomly generated according to a set of rules: 

- each phantom is 32x32x32 pixels and consists of 2 channels, one for bone and one for water.
- each phantom has a main, elliptical body centered in the image consisting of a low mass density of both water and bone
- The center of the body can be randomly offset by −3 to +3 pixels in both the x and y directions (uniform distribution). 
- The size of the main body varies, with the distance between the points on the axis varying between 1 and 3 pixels uniformly distributed. 
- each phantom has 2 internal features, represented as smaller elliptical regions within the main body.
- the bone and/or water density cannot be lower in a feature than the main body

Each feature represents a distinct elliptical region. Because photon counting detectors (PCDs) allow differentiating tissue types, we assign one feature a higher bone density and the other a higher water density. This design ensures that the network is trained on a task relevant to the clinical use of PCDs. 

Due to constraints within the iterative reconstruction algorithm, the bone and water densities of the features are restricted to be higher than those of the main body.

### acquiring data

A machine learning model will be trained to use information from sinogram space to adjust images in image space. The network will be trained to perform this conversion in iterative steps similar to an iterative reconstruction method [@mechlem2018joint]. 
First, simulations will be run on the randomly generated phantoms. For each phantom, detector measurements will be simulated for x-rays at multiple energy levels: ... kV, ... kV and ... kV. These measurements are done with 32 projections per phantom. An example phantom can be seen in [](#phantomSinogram), with a corresponding sinogram. (which energies?).

Next, each set of detector measurements will be reconstructed using an iterative algorithm with 40 iterations. These iterations will be converted to input-output pairs for training the model. Using a stride, the input is defined as the image from the nth iteration, and the corresponding output image is taken from the (n+stride)th iteration.

Using a stride between input and output images allows the model to learn from a wider variety of scenarios within the same training time. In this work, a stride of 4 is used. This work made use of ... phantoms, which generated ... input-output pairs using a stride of 4 .


### Attention based UNet

To enable the network to learn the correspondence between features in the image space and the projection (sinogram) space, we employ a UNet architecture with attention mechanisms integrated into the skip connections. The attention gates help the model to focus on which parts of the sinogram contribute to specific image regions.

This work uses a U-Net architecture with attention mechanisms added to the skip-connections as illustraded in [](#attunet). The architecture consists of five convolutional encoder blocks, a bottleneck block, and four decoder blocks. The number of channels increases through the encoder (64 → 128 → 256 → 512 → 1024), and then decreases symmetrically through the decoder. The output layer applies a sigmoid activation to produce voxel-wise probabilities.

The model is trained on an NVDIA Tesla A100, with 4 hours allocated per run. The model uses MSELoss as a loss function and Adam optimisation, which is a stochastic gradient descent method. [source] The learning rate is set to 0.001, which is commonly used with Adam optimisation. 

### shaping the data

A U-Net takes an image as an input and provides an image as an output. As 2 different images are used (image and projection) these images have to be appended into a single matrix. Aditionally, due to the pooling layer all dimensions must be powers of 16 as to not get an uneven amount of devisions. The projection matrix is 32 x 44 x 64 pixels, which we can pad with zeros to 32x48x64. The image will is 32 x 32 x 32 pixels which we can pad to 32 x 48 x 32 to fit the dimensions of the projections. The final input set to the network will be 10 x 32 x 48 x 96 as we have 10 channels as well. 

:::{figure} #iterations
:label: iteration
:width: 50%
:align: center
Set of iterations from the reconstruction of the bone image of a phantom using stride = 4. The phantom has 2 features with a higher concentration of bone. 
:::

// matching shapes 

The projection matrix will have the dimensions of the number of projections x y pixels detector x z pixels detector. The images during reconstruction refer to the intermediate images between each iteration of the iterative algorithm. As both parts of the data have a 3D format, the images will be padded with zeros to match the y-dimension of the projection matrix. The image and the projection are appended in the z-direction.







