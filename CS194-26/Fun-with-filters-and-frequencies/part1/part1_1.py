"""
Part 1.1 Finite Difference Operator
"""
import numpy as np
import skimage as sk
import skimage.io as skio
import matplotlib.pyplot as plt
from scipy.signal import convolve2d

# load image
img = skio.imread("cameraman.png", as_gray=True)
img = sk.img_as_float(img)

# filters in x and y directions
dx = np.array([[1, -1]])
dy = np.array([[1], [-1]])

# compute the gradient magnitude image
grad_x = convolve2d(img, dx, mode="same")
grad_y = convolve2d(img, dy, mode="same")
grad_mag = np.sqrt(grad_x**2 + grad_y**2)
grad_mag_thresh = np.where((grad_mag > 0.25), 1.0, 0.0)

# plot the results
plt.subplot(2, 2, 1)
plt.imshow(grad_x, cmap="gray")
plt.title("horizontal directional gradient")

plt.subplot(2, 2, 2)
plt.imshow(grad_y, cmap="gray")
plt.title("vertical directional gradient")

plt.subplot(2, 2, 3)
plt.imshow(grad_mag, cmap="gray")
plt.title("gradient magnitude")

plt.subplot(2, 2, 4)
plt.imshow(grad_mag_thresh, cmap="gray")
plt.title("gradient magnitude (threshold = 0.25)")

plt.show()
