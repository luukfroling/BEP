# results

## Visualisation reconstruction 

During evaluation the model has been run on 20 validation sets generated under the same conditions and rules as those used for training. By taking the first image from the iterative algorithm, 20 reconstructions can be made using the proposed AttU-Net model.  

:::{figure} #bone_reconstructions
:label: bone_recons
:align: center
Top row (water), left to right: Ground truth (GT) water image; iteration 0 (first reconstruction) from iterative algorithm; subsequent reconstructions by the model.
Bottom row (bone), left to right: GT bone image; iteration 0 (first reconstruction) from iterative algorithm; subsequent reconstructions by the model.
:::

The first 5 reconstructions of the second phantom are displayed in [](#bone_recons) and the first column displays the ground truth (GT) images as a reference. The rows show the reconstruction process using the model for water and bone respectively, where each column is an iteration. In the GT images, distinct features are visible: a bone structure in the lower left quadrant and a water structure in the top right quadrant. These serve as reference to evaluate the reconstruction quality.

The first reconstruction shows that the AttU-net model accuratly recongnises the key features of the phantom, however the water feature is faint in the water density image. As reconstruction progresses, the overall water density increases while the edges between distinct features of the phantom are blurred. The bone density image looks to converge to a state where the phantom has a lower density, with both water and bone features present.

## Visualisation comparison

To better understand the quality of the images compared to the iterative algorithm, the iterative algorithm has been used to reconstruct the phantoms as well using the same parameters as used during training. 

:::{figure} #plotPhantom
:label: reconPhantom
:align: center
(a) GT water image (b) smallest RMS water image iterative algorithm (c) water image from last iteration iterative algorithm (d) water image reconstructed by the model after 3 iterations (e) GT bone image (f) smallest RMS bone image iterative algorithm (g) bone image from last iteration iterative algorithm (h) bone image recontructed by the model after 3 iterations.
:::

[](#reconPhantom)  // relate it to previous image same phantom

Looking at the reconstructions corresponding to the smallest root mean square (RMS) error across all iterations (panel b and f), both water and bone features are present in their respective images. However the iterative algorithm does not fully distinguish between the two features, as the water feature can be faintly seen in the bone images and vice versa. For the final iteration of the iterative algorithm (panel c and g), the water image shows a clear reduction of the bone feature. This indicates an overcompensation compared to panel b.

The images reconstructed by the model after 3 iterations are shown in panel d and h. The water shows a nearly constant value throughout the entire phantom, unable to distringuish any features. The bone image clearly shows where the bone feature is located. 

## RMS over time

By calculating the root mean square (RMS) error at each iteration for both the iterative algorithm and the model with the ground truth (GT) image, we can compare reconstruction error as a function of time. 

:::{figure} #rmsTimePlot
:label: rmsTime
RMS error over time for three phantoms reconstructed using both the iterative method (orange: water; blue: bone) and the proposed model (red: water; green: bone).  
:::

[](#rmsTime) shows the RMS error over time for bone and water images reconstructed by both methods for 3 phantoms. The model has been cut off after the iterative algoritm stopped running for phantom 1 and 3. In all three cases, the model achieves a lower initial error but converges to a bigger error compared to the iterative algorithm. For both algorithms the RMS error of the water images converges to a higher value than the RMS of the bone images. 

:::{figure} #speedup_interpolation
:label: speedup_interpolated
The RMS-error plotted against time for 3 phantoms reconstructed using both the iterative method as well as the model. For both methods, the RMS is calculated for the bone and water image.  
:::

Looking at a dataset of 20 phantoms, the average time taken for the first iteration of the model is 9.3 seconds with a standard deviation of 1.5 seconds. The average speedup for bone for this step (how long it takes the iterative algorithm to get to the same RMS error) is 1.44 times, or 44% while the average speedup for water is 1.17 times, or 17%. As the average time per iteration has a standard deviation of 1.5 seconds, the average speedup can be plotted against iteration of the proposed model as illustrated in [](#speedup_interpolation). 



## wrong reconstructions

potentially display another set of reconstructions which are not good to highlight the inconsistency in the training data

