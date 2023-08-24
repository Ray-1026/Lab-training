import csv
import matplotlib.pyplot as plt
import numpy as np
import xml.etree.ElementTree as ET
import os
from skimage import io
from skimage.transform import resize

tree = ET.parse("ibug_300W_large_face_landmark_dataset/labels_ibug_300W_test_parsed.xml")
root = tree.getroot()
root_dir = "ibug_300W_large_face_landmark_dataset"

bboxes = []  # face bounding box used to crop the image
img_filenames = []  # the image names for the whole dataset

for filename in root[2]:
    img_filenames.append(os.path.join(root_dir, filename.attrib["file"]))
    box = filename[0].attrib
    # x, y for the top left corner of the box, w, h for box width and height
    bboxes.append([box["left"], box["top"], box["width"], box["height"]])

bboxes = np.array(bboxes).astype("float32")

with open("output.csv", newline="") as f:
    content = list(csv.reader(f))
    content.pop(0)
    row = 0

    for idx in range(16):
        img = io.imread(img_filenames[idx], as_gray=True)
        bbox = bboxes[idx]
        img = img[max(0, int(bbox[1])) : int(bbox[1] + bbox[3]), max(0, int(bbox[0])) : int(bbox[0] + bbox[2])]

        # scaling
        img = resize(img, (224, 224), anti_aliasing=True)

        points = np.zeros((68, 2))
        for j in range(136):
            points[j // 2][j % 2] = content[row][1]
            row += 1

        plt.subplot(4, 4, idx + 1)
        plt.imshow(img, cmap="gray")
        plt.scatter(points[:, 0], points[:, 1], c="red", s=5)
    plt.show()
