## Discussion

### non-machine learning speedup

- pytorch is highly optimized, mechlem is not
- 

### pixel size and number of projections necessary 

There will not be a linear relationship between the amount of projections necessary and the amount of pixels in the image. will this have an impact? 

### Model is only as good as your data

Looking at the accuracy of the model

### Adding cross attention between spaces

- adding cross attention can more accuratly predict which part of the sinogram attend to which parts of the image. Then convolution can still be used to reconstruct the image.