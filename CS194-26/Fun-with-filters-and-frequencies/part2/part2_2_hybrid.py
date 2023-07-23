"""
Part 2.2 Hybrid Images
"""
import cv2
import numpy as np
import skimage as sk
import skimage.io as skio
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from align_image import align_images


def hybrid_image(img1, img2, sigma1, sigma2):
    g1 = cv2.getGaussianKernel(31, sigma=sigma1)
    g1 = g1 * g1.T
    g2 = cv2.getGaussianKernel(41, sigma=sigma2)
    g2 = g2 * g2.T

    low_r = convolve2d(img1[:, :, 0], g1, mode="same")
    low_g = convolve2d(img1[:, :, 1], g1, mode="same")
    low_b = convolve2d(img1[:, :, 2], g1, mode="same")
    high_r = img2[:, :, 0] - convolve2d(img2[:, :, 0], g2, mode="same")
    high_g = img2[:, :, 1] - convolve2d(img2[:, :, 1], g2, mode="same")
    high_b = img2[:, :, 2] - convolve2d(img2[:, :, 2], g2, mode="same")

    low_img = np.stack([low_r, low_g, low_b], axis=2)
    high_img = np.stack([high_r, high_g, high_b], axis=2)

    plt.subplot(2, 2, 1)
    plt.imshow(np.log(np.abs(np.fft.fftshift(np.fft.fft2(sk.color.rgb2gray(img1))))), cmap="gray")
    plt.axis("off")
    plt.title("Original")

    plt.subplot(2, 2, 2)
    plt.imshow(np.log(np.abs(np.fft.fftshift(np.fft.fft2(sk.color.rgb2gray(low_img))))), cmap="gray")
    plt.axis("off")
    plt.title("Low-pass")

    plt.subplot(2, 2, 3)
    plt.imshow(np.log(np.abs(np.fft.fftshift(np.fft.fft2(sk.color.rgb2gray(img2))))), cmap="gray")
    plt.axis("off")
    plt.title("Original")

    plt.subplot(2, 2, 4)
    plt.imshow(np.log(np.abs(np.fft.fftshift(np.fft.fft2(sk.color.rgb2gray(high_img))))), cmap="gray")
    plt.axis("off")
    plt.title("High-pass")
    plt.show()

    return low_img + high_img


# Load images
img1 = sk.img_as_float(skio.imread("DerekPicture.jpg", as_gray="True"))
img1 = np.stack([img1] * 3, axis=2)
img2 = sk.img_as_float(skio.imread("nutmeg.jpg"))

# Align images
img2_aligned, img1_aligned = align_images(img2, img1)

# Hybrid iamges
sigma1 = 5
sigma2 = 17
hybrid = np.clip(hybrid_image(img1_aligned, img2_aligned, sigma1, sigma2), 0, 1)

# Result
skio.imshow(hybrid)
skio.show()


# My own images
# Load images
img3 = sk.img_as_float(skio.imread("dog.jpg", as_gray="True"))
img3 = np.stack([img3] * 3, axis=2)
img4 = sk.img_as_float(skio.imread("monkey.jpg"))

# Align images
img4_aligned, img3_aligned = align_images(img4, img3)

# Hybrid iamges
sigma3 = 3
sigma4 = 19
hybrid_2 = np.clip(hybrid_image(img3_aligned, img4_aligned, sigma3, sigma4), 0, 1)

# Result
skio.imshow(hybrid_2)
skio.show()


# save hybrid images
skio.imsave("res/hybrid_1.jpg", sk.img_as_ubyte(hybrid))
skio.imsave("res/hybrid_2.jpg", sk.img_as_ubyte(hybrid_2))
