import cv2
import json
import numpy as np
import skimage as sk
import skimage.io as skio
import matplotlib.pyplot as plt
from skimage.draw import polygon
from scipy.signal import convolve2d


def computeH(img1_pts, img2_pts):
    """
    solve linear equations Ah = b
    """
    N = len(img1_pts)
    A = np.zeros((2 * N, 8))
    b = np.zeros((2 * N, 1))

    # make matrices A and b
    for idx, pt in enumerate(img2_pts):
        A[idx * 2, 0:3] = pt[0], pt[1], 1
        A[idx * 2 + 1, 3:6] = pt[0], pt[1], 1
        A[idx * 2 : idx * 2 + 2, 6:] = -pt[0], -pt[1]

    for idx, pt in enumerate(img1_pts):
        A[idx * 2, 6:] = A[idx * 2, 6:] * pt[0]
        A[idx * 2 + 1, 6:] = A[idx * 2 + 1, 6:] * pt[1]
        b[idx * 2, 0] = pt[0]
        b[idx * 2 + 1, 0] = pt[1]

    # least square error
    h = np.linalg.lstsq(A, b, rcond=None)[0]
    h = np.vstack([h, [1]])
    return np.reshape(h, (3, 3))


def blendImage(canvas_1, canvas_2, canvas_full, mask_full, mask_1, mask_2, overlap):
    # Gaussian kernel
    gkernel = cv2.getGaussianKernel(89, 15)
    gkernel = gkernel @ gkernel.T

    # blur
    blur_mask_2 = np.dstack([mask_2 - overlap + overlap * convolve2d(mask_2, gkernel, mode="same")] * 3)
    blur_mask_1 = np.dstack([mask_1 - overlap + overlap * convolve2d(mask_1, gkernel, mode="same")] * 3)
    mask_full = np.dstack([mask_full] * 3)
    new_canvas = (mask_full - blur_mask_2) * canvas_1 + blur_mask_2 * canvas_full
    new_canvas = blur_mask_1 * new_canvas + (mask_full - blur_mask_1) * canvas_2

    plt.subplot(1, 2, 1)
    plt.imshow(canvas_full)
    plt.title("not use Gaussian filter")
    plt.subplot(1, 2, 2)
    plt.imshow(new_canvas)
    plt.title("use Gaussian filter")
    plt.show()

    return new_canvas


def warpImage(img1, img2, H):
    h1, w1, c = img1.shape
    h2, w2, _ = img2.shape
    H_inv = np.linalg.inv(H)

    # calculate the possible size of the new warped image
    bounds = np.array([[0, 0, 1], [0, h2, 1], [w2, 0, 1], [w2, h2, 1]])
    bounds = H @ bounds.T
    bounds[0:2, :] = np.round(bounds[0:2, :] / bounds[2, :])
    bounds = bounds.astype(int)
    max_y, min_y, max_x, min_x = np.max(bounds[1]), np.min(bounds[1]), np.max(bounds[0]), np.min(bounds[0])
    move_y, move_x = max(0, -min_y), max(0, -min_x)
    new_h, new_w = max(h1, h2, max_y) + move_y, max(w1, w2, max_x) + move_x

    # Homography : H * p = p_prime ==> p = H_inv * p_prime
    rr, cc = polygon([0, new_w, new_w, 0], [0, 0, new_h, new_h])
    p_prime = np.vstack([rr, cc, np.ones(len(rr))])
    p = H_inv @ p_prime
    p[0:2, :] = np.around(p[0:2, :] / p[2, :])
    p = p.astype(int)

    valid_pos = np.where((p[0, :] >= 0) & (p[1, :] >= 0) & (p[0, :] < w2) & (p[1, :] < h2))
    valid_x, valid_y = p[0, valid_pos], p[1, valid_pos]

    # canvas only image 1
    canvas_1 = np.zeros((new_h + 1, new_w + 1, c))
    canvas_1[move_y : h1 + move_y, move_x : w1 + move_x, :] = canvas_1[move_y : h1 + move_y, move_x : w1 + move_x, :] + img1

    # cavase only image 2
    canvas_2 = np.zeros((new_h + 1, new_w + 1, c))
    canvas_2[cc[valid_pos] + move_y, rr[valid_pos] + move_x, :] = img2[valid_y, valid_x, :]

    # canvas with image 1 and 2
    canvas_full = np.zeros((new_h + 1, new_w + 1, c))
    canvas_full[move_y : h1 + move_y, move_x : w1 + move_x, :] = canvas_full[move_y : h1 + move_y, move_x : w1 + move_x, :] + img1
    canvas_full[cc[valid_pos] + move_y, rr[valid_pos] + move_x, :] = img2[valid_y, valid_x, :]
    canvas_full[move_y : h1 + move_y, move_x : w1 + move_x, :] = (canvas_full[move_y : h1 + move_y, move_x : w1 + move_x, :] + img1) * 0.5

    # create different masks
    overlap = np.zeros((new_h + 1, new_w + 1))
    overlap[cc[valid_pos] + move_y, rr[valid_pos] + move_x] = overlap[cc[valid_pos] + move_y, rr[valid_pos] + move_x] + 0.5
    overlap[move_y : h1 + move_y, move_x : w1 + move_x] = overlap[move_y : h1 + move_y, move_x : w1 + move_x] + 0.5
    overlap = np.where(overlap < 1, 0, overlap)

    mask_full = np.zeros((new_h + 1, new_w + 1))
    mask_full[cc[valid_pos] + move_y, rr[valid_pos] + move_x] = 1
    mask_full[move_y : h1 + move_y, move_x : w1 + move_x] = 1

    mask_2 = np.zeros((new_h + 1, new_w + 1))
    mask_2[cc[valid_pos] + move_y, rr[valid_pos] + move_x] = 1

    # blending
    return blendImage(canvas_1, canvas_2, canvas_full, mask_full, mask_full - mask_2 + overlap, mask_2, overlap)


if __name__ == "__main__":
    img1 = sk.img_as_float(skio.imread("desk1.jpg"))
    img2 = sk.img_as_float(skio.imread("desk2.jpg"))

    with open("correspondences.json") as f:
        pts = json.load(f)

    img1_pts = pts["im1Points"]
    img2_pts = pts["im2Points"]

    H = computeH(img1_pts, img2_pts)

    imgwarped = warpImage(img1, img2, H)
    skio.imsave("res/desk.jpg", sk.img_as_ubyte(imgwarped))
