"""
Part 1.2 Derivative of Gaussian Filter
"""
import cv2
import numpy as np
import skimage as sk
import skimage.io as skio
import matplotlib.pyplot as plt
from scipy.signal import convolve2d

# loads image
img = skio.imread("cameraman.png", as_gray=True)
img = sk.img_as_float(img)

# x and y directions
dx = np.array([[1, -1]])
dy = np.array([[1], [-1]])

# Gaussian filter
gkernel = cv2.getGaussianKernel(7, sigma=1)
gkernel = gkernel * gkernel.T

# convolution
g_dx = convolve2d(gkernel, dx, mode="same")
g_dy = convolve2d(gkernel, dy, mode="same")
g_dxdy = convolve2d(g_dx, g_dy, mode="same")

grad_x = convolve2d(img, g_dx, mode="same")
grad_y = convolve2d(img, g_dy, mode="same")
grad_mag = np.sqrt(grad_x**2 + grad_y**2)
grad_mag_thresh = np.where((grad_mag > 0.1), 1.0, 0.0)

# plot
plt.subplot(1, 3, 1)
plt.imshow(g_dx, cmap="gray")

plt.subplot(1, 3, 2)
plt.imshow(g_dy, cmap="gray")

plt.subplot(1, 3, 3)
plt.imshow(g_dxdy, cmap="gray")

plt.show()

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
plt.title("gradient magnitude (threshold = 0.1)")

plt.show()
