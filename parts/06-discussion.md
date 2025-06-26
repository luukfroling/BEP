# Discussion
## Model is only as good as the data

The trained AttU-Net model reconstructed 20 images using a validation set of phantoms. The first four reconstructions of the second phantom are shown in [](#bone_recons) where the model successfully identifies features present in the phantom but fails to clearly distinguish between bone and water. 

Looking at the RMS error over time for the first three phantoms as plotted in [](#reconPhantom), reveals that the AttU-Net model initially achieves a lower RMS error compared to the iterative algorithm but eventually converges to a higher value.

When analyzing the final iterations of the iterative algorithm across all 20 validation phantoms, it can be seem some images do not converge to a visually accurate state as displayed in figure [](#wrong). Since the AttU-Net model was trained on these results, including those that failed to converge, this likely explains why the model itself also fails to converge effectively in later iterations. For future work, it is recommended to set an RMS error threshold and use only iterative reconstructions that meet this criterion for training.

## non-machine learning speedup

Looking at the software implementation for each method, the AttU-Net is more optimised than the iterative algorithm. The AttU-Net makes use of the PyTorch framework which is highly optimised for machine learning tasks. The iterative algorithm has been implemented from scratch and does not include any form of optimisations, which leads to an unfair comparison between the two algorithms. 

One way of optimising the iterative algorithm is parallelizing the computation of the SQS cost function. The SQS cost function is designed to look at the error relative to only it's neighbour, so it is part of the code most easily parallelised. 

## Image size, model size, number of projections

The model currently reconstructs images of size 32 × 32 × 32 pixels, which is too small for clinical applications. Future work should explore how performance and reconstruction time scale with increasing image size. Key factors to consider include:
- Number of projections: Larger images require more projections to fully capture all relevant information.
- Detector size: As image size increases, detector dimensions must also increase.
- Network size: To accommodate the added complexity of larger images, the network architecture must scale too.
- Hardware requirements: The current model already occupies 0.5 GB. Scaling up the network will demand additional storage and more powerful hardware. 

To properly compare running times, it is recommended to develop an AttU-Net-style model capable of reconstructing larger images to the same RMS error within a fixed number of iterations. This ensures that performance comparisons do not come at the cost of reduced reconstruction accuracy.

## Adding cross attention between spaces

Cross attention can implemented between two images as done by [@alaluf2023cross], where features from one image are used as a gating signal for an attention mechanism that processes the other image. Currently, two different spaces, image space and projection space, are added together in the same matrix. By using cross-attention, these spaces can be treated separately, with the attention mechanism serving as the interface between them. This allows for a more structured and interpretable flow of information between domains.

## Regularisation term

Since the AttU-Net is used to reconstruct CT scan images, its outputs must be clinically meaningful and physically plausible. To help guide the model toward producing more realistic reconstructions, a regularisation term can be introduced during training and reconstruction [@ge2023mb]. This term can incorporate physical constraints into the learning process, encouraging the model to generate outputs that not only appear correct but also align with the physical reality of CT imaging. T

For example, by penalising discrepancies between the forward projections of the reconstructed image and the actual measured sinogram data, the model is discouraged from producing outputs that deviate from what is physically observable. This helps the network learn to reconstruct features that are consistent with the measurement process, reducing the risk of hallucinated structures or anatomically implausible results.