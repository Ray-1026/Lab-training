"""
Part 2.1 Image "Sharpening"
"""
import cv2
import numpy as np
import skimage as sk
import skimage.io as skio
import matplotlib.pyplot as plt
from scipy.signal import convolve2d

# Load image
img = skio.imread("taj.jpg")
img = sk.img_as_float(img)

# Gaussian filter
gkernel = cv2.getGaussianKernel(7, sigma=1)
gkernel = gkernel * gkernel.T

# blur and sharpen
blurred = np.copy(img)

blurred[:, :, 0] = convolve2d(img[:, :, 0], gkernel, mode="same")
blurred[:, :, 1] = convolve2d(img[:, :, 1], gkernel, mode="same")
blurred[:, :, 2] = convolve2d(img[:, :, 2], gkernel, mode="same")

high_freq = img - blurred

# plot
plt.subplot(1, 4, 1)
plt.imshow(img, cmap="gray")
plt.title("a = 0")
plt.axis("off")

plt.subplot(1, 4, 2)
plt.imshow(np.clip(img + high_freq, 0, 1), cmap="gray")
plt.title("a = 1")
plt.axis("off")

plt.subplot(1, 4, 3)
plt.imshow(np.clip(img + 2 * high_freq, 0, 1), cmap="gray")
plt.title("a = 2")
plt.axis("off")

plt.subplot(1, 4, 4)
plt.imshow(np.clip(img + 3 * high_freq, 0, 1), cmap="gray")
plt.title("a = 3")
plt.axis("off")

plt.show()
