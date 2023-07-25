import os
import json
import imageio
import numpy as np
import skimage as sk
import skimage.io as skio
import matplotlib.pyplot as plt
from skimage.draw import polygon
from scipy.spatial import Delaunay


def computeAffine(tri1_pts, tri2_pts):
    # solve Ax=b
    homo_1 = np.hstack([tri1_pts, [[1], [1], [1]]])
    homo_2 = np.hstack([tri2_pts, [[1], [1], [1]]])
    return np.linalg.solve(homo_2, homo_1)


def bilinear_interpolation(img, res, homo, mapping):
    h, w = img.shape[0] - 1, img.shape[1] - 1
    homo = homo.astype(int)
    for i, j in zip(mapping, homo):
        xi, yi = int(i[0]), int(i[1])
        dx, dy = i[0] - xi, i[1] - yi
        res[j[1], j[0], 0] = (
            img[yi, xi, 0] * (1 - dx) * (1 - dy)
            + img[yi, min(xi + 1, w), 0] * (dx) * (1 - dy)
            + img[min(yi + 1, h), xi, 0] * (1 - dx) * (dy)
            + img[min(yi + 1, h), min(xi + 1, w), 0] * (dx) * (dy)
        )
        res[j[1], j[0], 1] = (
            img[yi, xi, 1] * (1 - dx) * (1 - dy)
            + img[yi, min(xi + 1, w), 1] * (dx) * (1 - dy)
            + img[min(yi + 1, h), xi, 1] * (1 - dx) * (dy)
            + img[min(yi + 1, h), min(xi + 1, w), 1] * (dx) * (dy)
        )
        res[j[1], j[0], 2] = (
            img[yi, xi, 2] * (1 - dx) * (1 - dy)
            + img[yi, min(xi + 1, w), 2] * (dx) * (1 - dy)
            + img[min(yi + 1, h), xi, 2] * (1 - dx) * (dy)
            + img[min(yi + 1, h), min(xi + 1, w), 2] * (dx) * (dy)
        )


def warp(img, src_pts, dst_pts, tri):
    res = np.zeros(img.shape)
    start_tri = src_pts[tri.simplices]
    end_tri = dst_pts[tri.simplices]

    for start, end in zip(start_tri, end_tri):
        # generate a mask of all pixels in the triangle
        canvas = np.zeros(img.shape)
        X = [i[0] for i in end]
        Y = [i[1] for i in end]
        rr, cc = polygon(X, Y)
        canvas[cc, rr] = 1
        mask = np.where(canvas)

        # inverse warp
        homo = np.dstack([mask[1], mask[0], np.ones(mask[0].shape)])[0]
        transform = computeAffine(start, end)
        mapping = homo @ transform
        bilinear_interpolation(img, res, homo, mapping)

    return np.clip(res, 0, 1)


def morph(img1, img2, pts1, pts2, warp_frac, dissolve_frac):
    warp_pts = (1 - warp_frac) * pts1 + warp_frac * pts2
    warp_tri = Delaunay(warp_pts)
    warp_1 = warp(img1, pts1, warp_pts, warp_tri)
    warp_2 = warp(img2, pts2, warp_pts, warp_tri)
    return (1 - dissolve_frac) * warp_1 + dissolve_frac * warp_2


def morph_sequence(img1, img2, pts1, pts2):
    frac = np.linspace(0, 1, 46)
    frames = []
    for f in frac:
        frames.append(morph(img1, img2, pts1, pts2, f, f))

    # make animation
    fps = 1 / 30
    with imageio.get_writer("res/morph.gif", mode="I", duration=fps, loop=0) as writer:
        for frame in frames:
            writer.append_data(sk.img_as_ubyte(frame))
        frames.reverse()
        for frame in frames:
            writer.append_data(sk.img_as_ubyte(frame))


def meanFace(directory):
    h, w = 480, 640
    filename = []
    all_points = []
    for fname in os.listdir(directory):
        # Image type : full frontal face, neutral expression, diffuse light
        if fname.endswith(".bmp"):
            # load points from .asf
            with open(f"{directory}/{fname[:5]}.asf") as asf:
                points = [[0, 0], [0, 479], [639, 0], [639, 479]]
                for i in asf:
                    i = i.split("\t")
                    if len(i) == 7 or len(i) == 10:
                        x, y = float(i[2]) * w, float(i[3]) * h
                        points.append([x, y])
                if len(points) != 4:
                    filename.append(fname)
                    all_points.append(points)

    all_points = np.array(all_points)
    avg_points = np.mean(all_points, 0)
    np.save("mean_face_points", avg_points)
    avg_tri = Delaunay(avg_points)
    warp_im = []

    for f, pt in zip(filename, all_points):
        im = sk.img_as_float(skio.imread(os.path.join(directory, f)))
        warp_im.append(warp(im, pt, avg_points, avg_tri))
        skio.imsave(f"res/averages/{f[0]}{f[1]}.jpg", sk.img_as_ubyte(warp_im[-1]))

    mean_face = np.mean(warp_im, 0)
    skio.imsave("res/mean_face.jpg", sk.img_as_ubyte(mean_face))
    return mean_face


if __name__ == "__main__":
    img1 = sk.img_as_float(skio.imread("funny_face.jpg"))
    img2 = sk.img_as_float(skio.imread("mr_bean.jpg"))

    # Part 1. Defining correspondences
    # Use tool from https://inst.eecs.berkeley.edu/~cs194-26/fa22/upload/files/proj3/cs194-26-aex/tool.html
    # to label correspondence points
    with open("points.json") as f:
        pts = json.load(f)

    pts1 = pts["im1Points"]
    pts2 = pts["im2Points"]
    for i in [[0, 0], [0, 239], [239, 0], [239, 239]]:
        pts1.append(i)
        pts2.append(i)
    pts1 = np.array(pts1)
    pts2 = np.array(pts2)
    tri1 = Delaunay(pts1)
    tri2 = Delaunay(pts2)

    plt.subplot(1, 2, 1)
    plt.imshow(img1)
    plt.triplot(pts1[:, 0], pts1[:, 1], tri1.simplices.copy())
    plt.plot(pts1[:, 0], pts1[:, 1], "or")
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.imshow(img2)
    plt.triplot(pts2[:, 0], pts2[:, 1], tri2.simplices.copy())
    plt.plot(pts2[:, 0], pts2[:, 1], "or")
    plt.axis("off")
    plt.show()

    # Part 2. Computing the "mid-way face"
    mid_pts = np.round(0.5 * pts1 + 0.5 * pts2)
    tri_mid = Delaunay(mid_pts)

    mid1 = warp(img1, pts1, mid_pts, tri_mid)
    mid2 = warp(img2, pts2, mid_pts, tri_mid)
    mid_face = 0.5 * mid1 + 0.5 * mid2

    plt.subplot(1, 3, 1)
    plt.imshow(img1)
    plt.axis("off")
    plt.subplot(1, 3, 2)
    plt.imshow(mid_face)
    plt.axis("off")
    plt.subplot(1, 3, 3)
    plt.imshow(img2)
    plt.axis("off")
    plt.show()

    skio.imsave("res/mid_face.jpg", sk.img_as_ubyte(mid_face))

    # Part 3. The morph sequence
    if not os.path.isfile("res/morph.gif"):
        morph_sequence(img1, img2, pts1, pts2)

    # Part 4. The "mean face" of a population
    directory = "face_data"
    if not os.path.isfile("res/mean_face.jpg"):
        mean_face = meanFace(directory)
    else:
        mean_face = sk.img_as_float(skio.imread("res/mean_face.jpg"))
        mean_points = np.load("mean_face_points.npy")

    ex = sk.img_as_float(skio.imread("ex.jpg"))
    ex_points = np.load("ex_points.npy")

    mean_tri = Delaunay(mean_points)
    ex_tri = Delaunay(ex_points)

    ex_to_mean = warp(ex, ex_points, mean_points, mean_tri)
    mean_to_ex = warp(mean_face, mean_points, ex_points, ex_tri)

    plt.subplot(1, 2, 1)
    plt.imshow(ex_to_mean)
    plt.title("warp into mean face")
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.imshow(mean_to_ex)
    plt.title("warp into example face")
    plt.axis("off")
    plt.show()

    skio.imsave("res/ex2mean.jpg", sk.img_as_ubyte(ex_to_mean))
    skio.imsave("res/mean2ex.jpg", sk.img_as_ubyte(mean_to_ex))

    # Part 5. Caricatures: Extrapolating from the mean
    num = 1
    alphas = [0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
    name = ["000", "025", "050", "075", "100", "125", "150", "175", "200"]

    for alpha in alphas:
        # formula ==> my_shape + alpha * (my_shape - average_shape)
        exag_points = ex_points + alpha * (ex_points - mean_points)
        out = warp(ex, ex_points, exag_points, Delaunay(exag_points))

        plt.subplot(3, 3, num)
        plt.imshow(out)
        plt.title(f"alpha={alpha}")
        plt.axis("off")

        skio.imsave(f"res/caricatures/{name[num-1]}.jpg", sk.img_as_ubyte(out))
        num += 1
    plt.show()
