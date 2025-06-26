# Conclusion

The proposed Attention U-Net (AttU-Net) model successfully recognises the outline of bone and water features within phantom images during density reconstruction. However, it fails to reliably distinguish between the two materials. This limitation is particularly relevant, as the ability to separate different materials, such as bone and water, is a key reason to use photon-counting detectors (PCDs) in imaging. 

Despite this, the model demonstrates significant promise. It achieves, in a single forward pass, the same reconstruction error that requires multiple iterations of a traditional algorithm; resulting in a 30% reduction in time to reach an equivalent RMS error. This highlights the model’s potential to accelerate, or even replace, conventional iterative techniques. While the model reaches this level quickly, it converges to a higher final RMS error than the iterative method it is tested against.

To move towards deploying the model in clinical settings, the model must be scaled to accommodate higher-resolution images. This will require increasing the network's capacity, such as increasing the amount of layers or increasing the amount of channels, to preserve details. Additionally, training data should be filtered by applying a threshold to the RMS error of the final iterative reconstructions, ensuring only high-quality examples are used for training. 
