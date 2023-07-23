"""
Part 2.4 Multiresolution Blending
"""
import cv2
import numpy as np
import skimage as sk
import skimage.io as skio
import matplotlib.pyplot as plt
from part2_3 import get_stack, combine_stacks

img1 = sk.img_as_float(skio.imread("funny_face.jpg"))
img2 = sk.img_as_float(skio.imread("trump.jpg"))
mask = sk.img_as_float(skio.imread("mask.jpg"))

gkernel = cv2.getGaussianKernel(41, sigma=9)
gkernel = gkernel * gkernel.T

img1_G, img1_L = get_stack(img1, gkernel)
img2_G, img2_L = get_stack(img2, gkernel)

img1_ = [img1_L[0], img1_L[2], img1_G[4]]
img2_ = [img2_L[0], img2_L[2], img2_G[4]]

l, r, combined = combine_stacks(img1_, img2_, gkernel, mask)
combined[3] = combined[3] - np.min(combined[3])
combined[3] = combined[3] / np.max(combined[3])

plt.plot()
plt.imshow(combined[3])
plt.show()

skio.imsave("res/funny_trump.jpg", sk.img_as_ubyte(combined[3]))
