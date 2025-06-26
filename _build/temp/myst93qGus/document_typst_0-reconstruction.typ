/* Written by MyST v1.3.26 */



```python
import pickle
import matplotlib.pyplot as plt

#load iteration.pkl
with open('iteration.pkl', 'rb') as f:
    x = pickle.load(f)

# make 2x4 plot with all images in x
plt.figure(figsize=(6, 3))
for i in range(8):
    plt.subplot(2, 4, i + 1)
    plt.imshow(x[i][0, :, 8:40, 16], cmap='gray')  # e.g., bone input
    plt.title(f"iteration {i+1}")
    plt.axis('off')
plt.tight_layout()
plt.show()
```

#image("files/47a8d42f599e527b323d72763db15f1e.png", width: 90%)