/* Written by MyST v1.3.26 */



= Attenuation coefficients water and bone

The attenuation coefficients change depending on the energy of the photons used.

```python
import scipy.io
import matplotlib.pyplot as plt

# Load the .mat file
mat_data = scipy.io.loadmat('libs/materialAttenuationBoneWater.mat')

first = mat_data['materialAttenuations'][:,0]
second = mat_data['materialAttenuations'][:,1]
plt.figure(figsize=(8, 6))
plt.plot(first, label='Bone')
plt.plot(second, label='Water')
plt.xlabel('Energy (keV)')
#nicely format the unit
plt.ylabel(r"Attenuation coefficient $\mu$(cm$^{-1}$)")
plt.legend()
plt.ylim(0,1)
plt.show()
```

#image("files/64ccdbd7840b97b269bc4a429597ebd6.png", width: 90%)