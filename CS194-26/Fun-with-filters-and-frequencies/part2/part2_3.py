"""
Part 2.3 Gaussian and Laplacian Stacks
"""
import cv2
import numpy as np
import skimage as sk
import skimage.io as skio
import matplotlib.pyplot as plt
from scipy.signal import convolve2d


# stack
def get_stack(img, gkernel, N=5):
    gaussian, laplacian = [], []
    temp = img.copy()

    for _ in range(N):
        r = convolve2d(temp[:, :, 0], gkernel, mode="same")
        g = convolve2d(temp[:, :, 1], gkernel, mode="same")
        b = convolve2d(temp[:, :, 2], gkernel, mode="same")
        blurred = np.stack([r, g, b], axis=2)
        gaussian.append(blurred)
        laplacian.append(temp - blurred)
        temp = blurred.copy()

        # normalize rgb
        laplacian[-1] = laplacian[-1] - np.min(laplacian[-1])
        laplacian[-1] = laplacian[-1] / np.max(laplacian[-1])

    return gaussian, laplacian


# combine stacks
def combine_stacks(img1, img2, gkernel, mask=[]):
    h, w, c = img1[0].shape
    if mask == []:
        mask = np.zeros((h, w, c))
        mask[:, w // 2 :, :] = 1

    overall, left, right = np.zeros((h, w, c)), np.zeros((h, w, c)), np.zeros((h, w, c))
    lhs, rhs, combined = [], [], []

    alpha_G, _ = get_stack(mask, gkernel)

    for i in range(0, 3):
        lhs.append(img1[i] * (1 - alpha_G[i * 2]))
        rhs.append(img2[i] * alpha_G[i * 2])
        combined.append(lhs[-1] + rhs[-1])
        left = left + lhs[-1]
        right = right + rhs[-1]
        overall = overall + combined[-1]
    lhs.append(left)
    rhs.append(right)
    combined.append(overall)
    return lhs, rhs, combined


if __name__ == "__main__":
    # Load image
    apple = sk.img_as_float(skio.imread("apple.jpeg"))
    orange = sk.img_as_float(skio.imread("orange.jpeg"))

    # Gaussian Kernel
    gkernel = cv2.getGaussianKernel(43, sigma=9)
    gkernel = gkernel * gkernel.T

    # Gaussain and Laplacian
    apple_G, apple_L = get_stack(apple, gkernel)
    orange_G, orange_L = get_stack(orange, gkernel)

    # plot results
    for i in range(5):
        plt.subplot(4, 5, i + 1)
        plt.imshow(apple_G[i])
        plt.axis("off")
        plt.subplot(4, 5, i + 6)
        plt.imshow(apple_L[i])
        plt.axis("off")
        plt.subplot(4, 5, i + 11)
        plt.imshow(orange_G[i])
        plt.axis("off")
        plt.subplot(4, 5, i + 16)
        plt.imshow(orange_L[i])
        plt.axis("off")
    plt.show()

    # combine stacks
    apple_ = [apple_L[0], apple_L[2], apple_G[4]]
    orange_ = [orange_L[0], orange_L[2], orange_G[4]]
    lhs, rhs, combined = combine_stacks(apple_, orange_, gkernel)

    # normalize rgb
    lhs[3] = lhs[3] - np.min(lhs[3])
    lhs[3] = lhs[3] / np.max(lhs[3])
    rhs[3] = rhs[3] - np.min(rhs[3])
    rhs[3] = rhs[3] / np.max(rhs[3])
    combined[3] = combined[3] - np.min(combined[3])
    combined[3] = combined[3] / np.max(combined[3])

    # plot
    for i in range(4):
        plt.subplot(4, 3, i * 3 + 1)
        plt.imshow(lhs[i])
        plt.axis("off")
        plt.subplot(4, 3, i * 3 + 2)
        plt.imshow(rhs[i])
        plt.axis("off")
        plt.subplot(4, 3, i * 3 + 3)
        plt.imshow(combined[i])
        plt.axis("off")
    plt.show()

    # save
    skio.imsave("res/oraple.jpeg", sk.img_as_ubyte(combined[-1]))
