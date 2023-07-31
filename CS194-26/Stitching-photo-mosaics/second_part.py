import os
import random
import numpy as np
import skimage as sk
import skimage.io as skio
import matplotlib.pyplot as plt
from skimage import transform
from harris import get_harris_corners, dist2
from first_part import computeH, blendImage, warpImage


def ANMS(h, coords, threshold=0.9, selected_num=500):
    """
    adaptive non-maximal suppression
    """
    radius = {}
    selected = []

    for i in coords:
        interest_points = []
        f_i = h[i[0], i[1]]
        for j in coords:
            f_j = h[j[0], j[1]]
            if f_i / f_j < threshold:
                interest_points.append(j)
        if interest_points:
            dist = dist2(np.array([i]), np.array(interest_points))
            radius[tuple(i)] = min(dist[0])

    radius = sorted(radius.items(), key=lambda x: x[1], reverse=True)

    # select top 500 points
    for i in range(selected_num):
        selected.append(list(radius[i][0]))
    return np.array(selected).T


def feature_descriptor(img, ip, patch=40):
    descriptor_vec = {}

    for i in ip:
        sample = img[i[0] - patch // 2 : i[0] + patch, i[1] - patch // 2 : i[1] + patch]
        downsample = transform.resize(sample, (8, 8), anti_aliasing=False)
        normalized = (downsample - np.mean(downsample)) / np.std(downsample)
        descriptor_vec[tuple(i)] = normalized.flatten().reshape(1, 64)

    return descriptor_vec


def feature_matching(descriptor_vec1, descriptor_vec2, threshold=0.5):
    matched = {}

    for ip1, vec1 in descriptor_vec1.items():
        dist = {}
        for ip2, vec2 in descriptor_vec2.items():
            dist[ip2] = np.sum((vec1 - vec2) ** 2)
        dist = sorted(dist.items(), key=lambda x: x[1])
        if dist[0][1] / dist[1][1] < threshold:
            matched[tuple([ip1[1], ip1[0]])] = tuple([dist[0][0][1], dist[0][0][0]])

    return matched


def ransac(matched_1, matched_2, threshold=0.8, iteration=1000):
    best_inliers = {}
    num_points = len(matched_1)

    for i in range(iteration):
        inliers = {}
        sample_1, sample_2 = [], []

        sample_indices = random.sample(range(num_points), 4)
        for idx in sample_indices:
            sample_1.append(matched_1[idx])
            sample_2.append(matched_2[idx])
        sample_1 = np.array(sample_1)
        sample_2 = np.array(sample_2)

        H = computeH(sample_2, sample_1)
        p_prime = H @ np.vstack([matched_1.T, np.ones(len(matched_1))])
        p_prime[0:2, :] = p_prime[0:2, :] / p_prime[2, :]
        loss = np.sqrt((p_prime[0, :] - matched_2.T[0, :]) ** 2 + (p_prime[1, :] - matched_2.T[1, :]) ** 2)

        for i in range(len(loss)):
            if loss[i] < threshold:
                inliers[tuple(matched_1[i])] = tuple(matched_2[i])
        if len(inliers) > len(best_inliers):
            best_inliers = inliers.copy()

    return best_inliers


if __name__ == "__main__":
    img1 = sk.img_as_float(skio.imread("desk1.jpg"))
    img2 = sk.img_as_float(skio.imread("desk2.jpg"))
    img1_gray = sk.img_as_float(skio.imread("desk1.jpg", as_gray=True))
    img2_gray = sk.img_as_float(skio.imread("desk2.jpg", as_gray=True))

    # Harris corner
    h1, coords_1 = get_harris_corners(img1_gray)
    h2, coords_2 = get_harris_corners(img2_gray)

    plt.subplot(1, 2, 1)
    plt.imshow(img1)
    plt.scatter(coords_1[1], coords_1[0], s=1, c="red")
    plt.title("desk1.jpg")
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.imshow(img2)
    plt.scatter(coords_2[1], coords_2[0], s=1, c="red")
    plt.title("desk2.jpg")
    plt.axis("off")
    plt.show()

    # adaptive non-maximal suppression
    if os.path.isfile("interest_points_1.npy") and os.path.isfile("interest_points_2.npy"):
        ip_1 = np.load("interest_points_1.npy")
        ip_2 = np.load("interest_points_2.npy")
    else:
        ip_1 = ANMS(h1, coords_1.T)
        ip_2 = ANMS(h2, coords_2.T)
        np.save("interest_points_1", ip_1)
        np.save("interest_points_2", ip_2)

    plt.subplot(1, 2, 1)
    plt.imshow(img1)
    plt.scatter(ip_1[1], ip_1[0], s=1, c="red")
    plt.title("desk1.jpg")
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.imshow(img2)
    plt.scatter(ip_2[1], ip_2[0], s=1, c="red")
    plt.title("desk2.jpg")
    plt.axis("off")
    plt.show()

    # feature descriptor extraction
    descriptor_vec1 = feature_descriptor(img1_gray, ip_1.T)
    descriptor_vec2 = feature_descriptor(img2_gray, ip_2.T)

    # feature matching
    matched = feature_matching(descriptor_vec1, descriptor_vec2)
    matched_1 = np.array(list(matched.keys())).T
    matched_2 = np.array(list(matched.values())).T

    plt.subplot(1, 2, 1)
    plt.imshow(img1)
    plt.scatter(matched_1[0], matched_1[1], s=5, c="blue")
    plt.subplot(1, 2, 2)
    plt.imshow(img2)
    plt.scatter(matched_2[0], matched_2[1], s=5, c="blue")
    plt.show()

    # Random Sample Consensus
    best_inliers = ransac(matched_1.T, matched_2.T)
    best_inliers_1 = np.array(list(best_inliers.keys())).T
    best_inliers_2 = np.array(list(best_inliers.values())).T

    plt.subplot(1, 2, 1)
    plt.imshow(img1)
    plt.scatter(best_inliers_1[0], best_inliers_1[1], s=5, c="blue")
    plt.subplot(1, 2, 2)
    plt.imshow(img2)
    plt.scatter(best_inliers_2[0], best_inliers_2[1], s=5, c="blue")
    plt.show()

    H = computeH(best_inliers_1.T, best_inliers_2.T)
    imgwarped = warpImage(img1, img2, H)

    skio.imsave("res/desk_autostitching.jpg", sk.img_as_ubyte(imgwarped))
