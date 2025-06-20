## results

### speedup reconstruction

During evaluation the model has been run on 20 validation sets constructed with the same rules as the training data. By calculating the root mean square (RMS) error at each iteration for both the iterative algorithm and the model with the ground truth (GT) image, we can compare reconstruction error as a function of time. 

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

### Visualise reconstruction

To further understand the results we can look at a specific reconstruction. 

:::{figure} #plotPhantom
:label: reconPhantom
:align: center
(a) GT water image (b) smallest RMS water image iterative algorithm (c) water image from last iteration iterative algorithm (d) water image reconstructed by the model after 3 iterations (e) GT bone image (f) smallest RMS bone image iterative algorithm (g) bone image from last iteration iterative algorithm (h) bone image recontructed by the model after 3 iterations.
:::

[](#reconPhantom) shows the GT for water and bone and the corresponding reconstructions. Aside from the final reconstruction it is important to look at the reconstruction with the lowest error of the iterative approach. The ground truth (GT) consists of a bone feature in the lower left part of the image and a water feature in the top right part of the image. Starting with water, the most accurate reconstruction shows signs of both features. The last iteration however shows a larger bone density for the water feature but the algortithm seems to have overcompensated by fully removing the water density for the feature in the bottom left.

### wrong reconstructions

potentially display another set of reconstructions which are not good to highlight the inconsistency in the training data

