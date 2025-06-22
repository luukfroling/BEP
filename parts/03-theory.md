# Theory

## Basics CT scans 

A computed tomography (CT) scanner is an imaging device that uses x-rays to create cross-sectional images of an object. It consists of an x-ray source and a detector placed at a distance from eachother. During scanning, both the source and the detector will rotate around the object to be imaged in steps. For each step, the source emits x-rays that pass through the object; the transmitted photons are measured by the detector. The number of detected photons at position i along the detector is given by:

```{math}
:label: first 
\hat{y}_i  = N_0 \cdot \exp\left(-\int \mu(x, E) \, dx\right)
```

where $N_0$ is the number of the x-rays emitted by the source, $\mu(x, E)$ is the linear attenuation coefficient and the integral in the exponent represents the cumulative attenuation along the x-ray path. $\mu(x, E)$ describes how many of the incoming x-rays are absorbed per unit of length, which depends on both the material of the object and the energy of the x-ray [@kamalian2016ct_principles]. 

By rotating the source and detector around the object and recording measurements at multiple angles, a sinogram can be created. A sinogram is a 2D plot where each row corresponds to a detector reading at a specific projection angle.

:::{figure} #phantomAndSinogram
:label: phantomSinogram
:figclass: H
*left:* a 32x32 pixel phantom representing varying bone densities. *right* corresponding sinogram with 64 projection angles. The lighter regions in the sinogram indicate higher photon counts. The CT energy used was [TODO].
:::

A phantom and a corresponding sinogram can be seen in [](#phantomSinogram). The phantom is a 32x32 image, where each pixel represents a part of the phantom with a specific bone density. Lighter coloured areas having a higher bone density where darker areas have a lower bone density. On the right is the corresponding sinogram which consists of 64 measurements of $\hat{y}_i$ at different angles. The lighter parts of the sinogram correspond to more detected photons. 

## Cone beam CT scan

The sinogram in [](#phantomSinogram) is true for a 1D detector, where we construct a single slice. A cone beam CT scan uses a source emitting x-rays in a cone-beam shape, which can be detected on a 2D flat-panel detector [@venkatesh2017cbct]. 

:::{figure} parts\Figures\cone_beam_ct.png
:label: cone_beam
:width: 50%
:figclass: H

A schematic drawing of a cone-beam CT scan: (a) the 2D detector panel, (b) the phantom, (c) rotation direction of the source-detector pair (d) the x-ray source. 
:::

[](#cone_beam) shows a cone beam CT scan setup, where both the detector and the source rotate around the object to acquire a set of 2D projection images at multiple angles. these measurements are known as rays. 

The relationship between the object and the measured rays can be described by a projection matrix $A$. This matrix has dimensions $ M \times N $ where $N$ is the number of voxels in the object and $ M $ is the number of rays (measurements) collected by the detector. Each element $ a_{mn} $ of the matrix is the contribution of the nth voxel of the object to be imaged to the mth ray of the detector [@Yang2017ProjectionMatrix]. The measured projections can thus be modeled as:

```{math}
:label: projection_eq
y = Ax
```

where $ x $ is the vectorised representation of the voxel values and $ y $ contains the corresponding ray measurements. This formulation is will be used by the reconstruction algorithms during the simulations.


## Detector types

Current CT scanners use an energy integrating detector (EID) [@marth2023photon]. An EID measures the intensity of the incoming x-rays by a two-step process: scintillation followed by photodetection. The scintillator will absorb the x-rays which have passed through the body and emit light in the visible spectrum. These re-emitted photons can be detected by photodetectors located underneath the scintillator. Due to the difference in energy of the incoming x-ray and the emitted visible light, multiple photons can be re-emitted. To measure a signal, EIDs integrate over time which loses all energy dependent information [@taguchi2013photon].    

:::{figure} Figures/detectorTypes.png
:label: detectors
:width: 50%

*left:* An EID where the incoming x-rays are absorbed within the scintillator. Several photons within the visible light spectrum are re-emitted which can be measured by the photodiodes located below the scintillator. *right:* A PCD where incoming x-rays create electron-hole pairs in a semiconductor. This creates a current between the top and bottom of the semiconductor which can be measured by electrodes, preserving energy dependent information.
:::

Photon-counting detector (PCD) CT scans improve the measurements by directly transforming the incoming CT scans into an energy-dependent signal which can be measured as illustrated in [](#detectors). The incoming x-rays will be absorbed in a semiconductor layer, forming electron-hole pairs which generate an electric signal proportional to the energy of the absorbed photon. Unlike EIDs, this allows the detector to retain energy-dependent information for each photon. 

:::{figure} #materialAttenuation
:label: matatt
:width: 10%
The linear attenuation coefficient ($\mu$) of bone (blue) and water (orange) plotted as a function of the incoming x-ray photon energy. The different energy-dependent attenuation behaviour enables material differentiation in photon-counting CT.
:::

The linear attenuation coefficient used in equation [](#first) depends on the material and the x-ray energy used as illustrated by the attenuation curves in [](#matatt). Using a source emitting x-rays at a range of energies, energy bins can be generated by the PCD where each bin contains energy-specific measurements [@pcd2022]. This enables the differentiation between materials with similar densities which would otherwise appear similar in conventional CT scans using EIDs.


## Material decomposition 

Since different materials attenuate x-rays differently across the energy spectrum, PCD-CT can be used to distinguish between different tissue types within the body. To simulate this process and to describe the attenuation properties of different materials in a phantom, material decomposition is used [@mechlem2018joint]. In this work, material decomposition models each material in the phantom as a linear combination of two basis materials: bone and water.

The reconstructed image of a PCD CT scan will therefore consist of 2 images, one displaying the bone density and one displaying water density. The linear attenuation coëfficient of any material within the phantom can then be described as:

```{math}
:label: muE
\mu(E) = \sum_{b=1}^{B} A_b f_b(E)
```
where $A_b$ is the line integral of material b and $f^b(E) $ describes the attenuation coefficient of the basis material b. Combining equation [](#first) and [](#muE) gives an expression for the photon count at the detector:

```{math}
\hat{y}_i = \int_0^{\infty} \phi_{\text{eff},i}(E) \exp\left( - \sum_{b=1}^{B} A_b^i f_b(E) \right) \, dE 
```

where $ \phi_{\text{eff},i}(E) $ is an effictive x-ray spectrum which includes all source and detector effects. 

## Projection algorithms 

To view the cross-sectional image of an object from these measurements, either from a detector or from simulations using equation [](#projection_eq), a reconstruction algorithm needs to be used. The simplest form of a reconstruction algorithm is a backprojection algorithm; during reconstruction, the algorithm spreads each projection back across the image plane along the same angle it was acquired. This often leads to blurry images as the intensity is spread along lines rather than concentrated at specific points [@zeng2001image]. Backprojection works primarily for single energy CT scans as illustrated in [](#phantomSinogram), however it can be used for dual-energy CT scans by a two-step image based method. The first step is to reconstruct an image for each energy bin and these intermediate images are then decomposed into material-dependent images [@mory2018comparison]. 

The two-step, image based method has several drawbacks. First, beam hardening artifacts often occur as the attenuation coefficient is still averaged over a line. Secondly, the first step leads to a loss of information as there is no one-to-one mapping between the projections and the images. The second step is unable to compensate for this loss as it has no access to the photon counts. Recently one-step methods have been proposed which reconstruct material-specific images directly from photon counts [@mechlem2018joint]. These are all iterative methods as no analytical inversion formula exists. 

Import parameters of the interative algorithm introduced are the data attachment term and the cost regularisation. The data attachment term ensures the reconstructed material images explain the measured photon counts and the regularisation term enforces smoothness or prior knowledge on the material images [@Zhang2018Regularization], a cost function can be defined which includes both terms. The iterative algorithm reconstructs an image for each material and compares the image to the measurements using the cost function and updates the images accordingly. This is computationally heavy and 

## Machine learning - Neural networks

Neural networks are a subset of machine learning and play a crucial role in deep learning algorithms. A simple, fully connected neural network consists of nodes stacked in layers: an input layer, one or more hidden layers and an output layer.

:::{figure} Figures\neuralnetwork.png
:label: neural_network
A neural network consisting of 3 input nodes, 2 hidden layers with 4 nodes each and an output layer with 3 nodes. The input values are represented by an array x, the hidden layers by array h' and h'' and the output layer by y.
:::

All nodes in a layer are connected to an adjacent layer by weigths. [](#neural_network) shows a simple neural network, where input layer x and first hidden layer h' are connected by a matrix of weights $W_1$, the states of the first hidden layer can be calculated as 

```{math}
:label: denseLayer
\mathbf{h}_1 = \sigma(\mathbf{W}_1 \mathbf{x} + \mathbf{b}_1)
```
with $\mathbf{W}_{1,nm}$ representing the weight connecting the m-th input neuron to the n-th neuron in the first hidden layer, and σ is an activation function (e.g., sigmoid or softmax) that scales the output between 0 and 1 to stabilise the network. Equation [](#denselayer) can be repeated for every following layer where the output of a layer is used as an input to calculate the following layer. 

Neural networks learn by backpropagation. Backpropagation looks at the error between the output of the network and the desired output. This error is generally propogated backwards through the network by looking at how each weight needs to change to minimise this error. 

## Machine learning - Convolutional neural network

A convolutional neural network (CNN) can be used to more accuratly extract features from images for processing. A CNN consist of a kernel (also called a filter) and one or more channels. The kernel is smaller than the input data but has the same number of dimensions (e.g., 2D for 2D images). The kernel contains weights that are learned during training.

The kernel is moved across the input data, performing element-wise multiplication with the region of the input it covers, followed by a sum. This operation extract features such as edges, textures and shapes [@yamashita2018convolutional]. We can write the output feature map of a convolutional layer as 

```{math}
S(i, j) = (\mathbf{K} * \mathbf{X})(i, j) = \sum_m \sum_n \mathbf{K}(m, n) \cdot \mathbf{X}(i + m, j + n)
```

where $ (i,j) $ are spatial indices, $\mathbf{X}$ the relevant kernel and m and n the kernel sizes. After a convolution layer, $S(i, j)$ can be either vectorised to be used as an input for a fully connected layer as described in equation [](#denseLayer) or used as an input for a second convolution layer. 

:::{figure} #convolutionExample
:label:convex
*left :* original phantom with a 2D 3x3 randomly initialised kernel drawn on top. each individual square represents a pixel, with the outer square representing the kernel. *right :* phantom after 2D convolution layer has been applied. 
:::

The left part of [](#convex) shows a 3x3 convolution kernel (red) placed on a phantom and the right part of the figure shows the imgage of the phantom after the convolution. Even with randomly initialised weights, there is already an edge-detection like pattern. 

A CNN typically consists of multiple layers stacked on top of each other, allowing deeper feature extraction. Early layers can detect simple features like edges, while deeper layers can detect more complex structures like shapes and objects. 

## Machine learning - U-Net architecture

Multiple convolution layers can be combined to form an encoder. An encoder reduces the size of the image between each layer using a max pool layer.A max pool layer devides the input into rectangular regions and taks the maximum value from each region, reducing the image size and allowing the network to capture more abstract and higher-level data.

Similarly a decoder can be used to an abstract representation and generate an output image. A decoder uses deconvolution layers which increases the size of the image as more features are added back in. 

:::{figure} Figures\Unet.png
:label: unet
:width: 75%
A U-Net architecture for an image with a 32x32 lowest possible resolution. Each blue square represents a multi-channel feature map with the amount of channels denoted on top of the box. Gray arrows represent cross over layers with the white boxes representing copied layers from the encoder *Figure reproduced from* [@oktay2018attention]
:::

[](#unet) shows a U-Net architecture, which combines an encoder and a decoder with an equal amount of layers. Between each layer of the encoder and decoder part of the network, there is a skip-connection layer as well which copies the feature map from the encoder layer to the decoder layer. This allows the decoder to use features which might have been lost while encoding the image. 


## Machine learning - Attention gate

For feature extraction from images where both local and non-local features are of importance, convolution layers can be combined with attention mechanisms. An attention mechanism is a technique used in machine learning used to allow machine learning models to attend to the most relevant parts of the input data [@vaswani2017attention]. 

An attention mechanism takes the input vector $x_{i}^{l}$ and multiplies it by an attention score $\alpha_{i}^{l}$ to perserve only the activations relevant to the task. To calculate $\alpha_{i}^{l}$,  an intermediate attention score $q_{\text{att}}^{l}$ (the attention score $q_{\text{att}}$ for layer l) can be defined as 
```{math}
q_{\text{att}}^{l} = \psi^{T} \, \sigma \left( W_{x}^{T} x_{i}^{l} + W_{g}^{T} g_{i} + b_{g} \right) + b_{\psi}
```
where $g_{i} $ the gating feature vector, $ W_{x}^{T} $ and $ W_{g}^{T} $ weight matrices similar to those used in equation [](#denseLayer), $ \psi^{T} $ another linear transformation and $ \sigma $ an activation function. The final attention coefficient $ \alpha_{i}^{l} $, the final attention mask that says which parts of the feature map should be passed forward, is then calculated as 

```{math}
\alpha_{i}^{l} = \sigma_{2} \left( q_{\text{att}}^{l}(x_{i}^{l}, g_{i}; \Theta_{\text{att}}) \right)
```
with \Theta_{\text{att}} representing the learned parameters of the attention block: all weights and biases. The final output of the attention gate it 

```{math} 
\tilde{x}_i^l = \alpha_i^l \cdot x_i^l
```

where regions get $\alpha = 1$ where less important regions get $\alpha \approx 0$  


## Machine learning - Attention U-Net

Attention mechanisms can be added to the skip-connections of the U-Net model in [](#unet), allowing the network to focus on non-local features during the reconstruction of the image. The combination of attention mechanism and convolutional layers can be used to recognise certain features like edges and to make connection between features in different parts of the image [@oktay2018attentionunet].

:::{figure} Figures\AttU-Net.png
:label: attunet
Schematic overview of a U-Net model with attention gates added to the skip-connections. The inset (top) shows a zoomed-in view of an attention gate, the gating signal $g$ is taken from the previous decoding layer and the input $x^l$ is the skip-connection layer. *Figure reproduced from* [@oktay2018attentionunet]
:::

The gate vector $g$, also called the gating signal, for each skip-connection layer connects the output of the previous decoder layer to the skip-connection input vector. This is illustrated in [](#attunet). //Work in progress

For each skip connection the output of the previous decoder layer is used as a gating signal $g$. //Work in progress 

:::{figure} #show_attention_overlay
:label: attention_network

An attention map showing how an attention mechanism can be used to have the model focus more on important features, which in this case is the sinogram.
:::

[](#attention_network) shows an attention map from a AttU-Net model trained on CT scan reconstruction. An attention map has been drawn to illustrate $\alpha_{i}^{l}$ for the first skip-connection. For this specific application it is as expected the attention gate focuses on the lower-intensity parts of the sinogram as these areas provide the most information about the imaged object.  



