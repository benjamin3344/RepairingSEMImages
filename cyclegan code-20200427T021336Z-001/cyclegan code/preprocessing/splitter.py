# splitter.py
# Author: Raymond Zeng
# April 13, 2020
#
# Splits 600 x 1000 images into 200 x 333 images (omitting the last row)

import numpy as np

# constants
source_file = "output.npy"
dest_file = "output_small.npy"

data = np.load(source_file)
images = []
for image in data:
    for i in range(0, 600, 200):
        for j in range(0, 999, 333):
            image_small = image[i:i + 200, j:j + 333, :]
            images.append(image_small)

images = np.asarray(images, dtype = np.uint8)
images = images.reshape([-1, 200, 333, 1])
np.save(dest_file, images)
