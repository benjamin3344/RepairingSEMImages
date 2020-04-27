# noise.py
# Author: Raymond Zeng
# April 11, 2020
#
# Adds background noise and simulates charging particles in grayscale SEM images
#
# Code for Perlin noise and fractals functions (lines 54-87) from
# https://pvigier.github.io/2018/06/13/perlin-noise-numpy.html
#
# Code for contours (lines 124-154) from tutorial at https://www.pyimagesearch.com/
# 2016/10/31/detecting-multiple-bright-spots-in-an-image-with-python-and-opencv/

from imutils import contours
from skimage import measure
import numpy as np
import imutils
import cv2
import random
import math

# constants
source_file         = "data.npy"    # source file path for images in .npy file format
dest_file           = "output.npy"  # destination file path for images in .npy file format
image_size          = (600, 1000)   # rows, columns
contrast            = True          # add contrast to simulate charging particles
sparks              = True          # add sparks to simulate charging particles
perlin              = True          # add Perlin noise to simulate noisy background
sine_freq           = 3             # frequency of sine map (for adding contrast)
sine_amp            = 0.2           # amplitude of sine map
sine_strength       = 0.5           # strength of sine map
perlin_freq         = 10            # frequency of Perlin noise map
perlin_octaves      = 2             # octaves for Perlin noise map
perlin_amp          = 0.2           # amplitude of Perlin noise map
perlin_strength     = 0.1           # strength of Perlin noise map
min_particle_size   = 300           # minimum particle size detected
contrast_strength   = 1.25          # strength of contrast map
particle_prob       = 0.6           # probability of a particle getting sparks added
point_prob          = 0.2           # probability of a contour point being used to draw spark
mu_dist             = 0.8           # mean fraction of distance from center of particle
                                    # to contour point for drawing beginning of spark
sigma_dist          = 0.1           # variance of distance from particle center to spark
min_spark_dist      = 0.6           # minimum distance from particle center to spark
max_spark_dist      = 0.9           # maximum distance from particle center to spark
min_spark_br        = 200           # minimum brightness of pixel required to draw spark from
mu_length           = 7             # mean spark length
sigma_length        = 5             # variance of spark length
mu_rotation         = 0             # mean rotation (in degrees) for spark
sigma_rotation      = 5             # variance of rotation (in degrees) for spark
mu_thickness        = 2             # mean thickness of spark
sigma_thickness     = 0.5           # variance in thickness of spark
min_thickness       = 1             # minimum thickness of spark
max_thickness       = 3             # maximum thickness of spark

def generate_perlin_noise_2d(shape, res):
    def f(t):
        return 6*t**5 - 15*t**4 + 10*t**3

    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])
    grid = np.mgrid[0:res[0]:delta[0],0:res[1]:delta[1]].transpose(1, 2, 0) % 1
    # Gradients
    angles = 2*np.pi*np.random.rand(res[0]+1, res[1]+1)
    gradients = np.dstack((np.cos(angles), np.sin(angles)))
    g00 = gradients[0:-1,0:-1].repeat(d[0], 0).repeat(d[1], 1)
    g10 = gradients[1:,0:-1].repeat(d[0], 0).repeat(d[1], 1)
    g01 = gradients[0:-1,1:].repeat(d[0], 0).repeat(d[1], 1)
    g11 = gradients[1:,1:].repeat(d[0], 0).repeat(d[1], 1)
    # Ramps
    n00 = np.sum(grid * g00, 2)
    n10 = np.sum(np.dstack((grid[:,:,0]-1, grid[:,:,1])) * g10, 2)
    n01 = np.sum(np.dstack((grid[:,:,0], grid[:,:,1]-1)) * g01, 2)
    n11 = np.sum(np.dstack((grid[:,:,0]-1, grid[:,:,1]-1)) * g11, 2)
    # Interpolation
    t = f(grid)
    n0 = n00*(1-t[:,:,0]) + t[:,:,0]*n10
    n1 = n01*(1-t[:,:,0]) + t[:,:,0]*n11
    return np.sqrt(2)*((1-t[:,:,1])*n0 + t[:,:,1]*n1)

def generate_fractal_noise_2d(shape, res, octaves, persistence=0.5):
    noise = np.zeros(shape)
    frequency = 1
    amplitude = 1
    for _ in range(octaves):
        noise += amplitude * generate_perlin_noise_2d(shape, (frequency*res[0], frequency*res[1]))
        frequency *= 2
        amplitude *= persistence
    return noise

# load data
train_data = np.load(source_file)

# make sine map
axis_0 = np.arange(image_size[0])
axis_1 = np.arange(image_size[1])
axis_0 = np.sin(axis_0 * np.pi / 180. * sine_freq) * sine_amp + sine_strength
axis_1 = np.cos(axis_1 * np.pi / 180. * sine_freq) * sine_amp + sine_strength
sin_arr = np.empty([0, image_size[1]])
for (i, n) in enumerate(axis_0):
    sin_arr = np.append(sin_arr, [np.add(n, axis_1)], axis = 0)

# generate Perlin noise map
if perlin:
    perlin_map = generate_fractal_noise_2d((image_size[0], image_size[1]),
        (int(image_size[0] // perlin_freq), int(image_size[1] // perlin_freq)), perlin_octaves)
    perlin_map = np.multiply(perlin_map, perlin_amp)
    perlin_map = np.add(perlin_map, perlin_strength)
    perlin_map = np.round_(np.multiply(perlin_map, 255))
else:
    perlin_map = np.zeros((image_size[0], image_size[1]))

# create empty image array to append images to
image_arr = np.empty([0, image_size[0], image_size[1], 1])
image_arr = image_arr.astype("uint8")

for i in range(train_data.shape[0]):
    # print progress
    if (i > 0 and i % 10 == 0):
        print("Processed", i, "images...")

    img = train_data[i]
    img = img.astype("uint8")
    img = img.reshape(image_size[0], image_size[1], 1)

    # perform a Gaussian blur
    blurred_img = cv2.GaussianBlur(img, (11, 11), 0)

    # binarized blurred image using Otsu's method
    thre_img = cv2.threshold(blurred_img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]

    # remove specks
    thre_img = cv2.erode(thre_img, None, iterations=2)
    thre_img = cv2.dilate(thre_img, None, iterations=4)

    labels = measure.label(thre_img, connectivity=2, background=0)
    mask = np.zeros(thre_img.shape, dtype="uint8")

    # loop over the unique components
    for label in np.unique(labels):
    	# if this is the background label, ignore it
    	if label == 0:
    		continue
    	# otherwise, construct the label mask and count the number of pixels
    	labelMask = np.zeros(thre_img.shape, dtype="uint8")
    	labelMask[labels == label] = 255
    	numPixels = cv2.countNonZero(labelMask)
    	# if the number of pixels in the component is sufficiently
    	# large, then add it to our mask of "large blobs"
    	if numPixels > min_particle_size:
    		mask = cv2.add(mask, labelMask)

    # find the contours in the mask, then sort them from left to right
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = contours.sort_contours(cnts)[0]

    img_copy = np.copy(img) # make copy of image for adding noise
    img_copy = img_copy.reshape(image_size[0], image_size[1])

    # add contrast to the image to simulate charging particles
    if contrast:
        # increase contrast of masked regions
        contrast_map = np.copy(mask)
        contrast_map = contrast_map / 1000 * contrast_strength + 1
        contrast_map = np.multiply(contrast_map, np.where(contrast_map > 1, \
            sin_arr, 1))
        img_copy = np.multiply(img_copy, contrast_map)

    # draw sparks to simulate charging
    spark_map = np.zeros((image_size[0], image_size[1]))
    for c in cnts:
        ran1 = random.random()
        if sparks and ran1 < particle_prob:
            # find center of particle
            M = cv2.moments(c)
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

            for i in range(c.shape[0]):
                ran2 = random.random()
                if ran2 < point_prob:
                    # find vector from center to contour point
                    point = (c[i, 0, 0], c[i, 0, 1])
                    vec_end = tuple(np.subtract(point, center))

                    # find start point of spark
                    dist_from_center = np.inf
                    while dist_from_center < min_spark_dist or \
                        dist_from_center > max_spark_dist:
                        dist_from_center = np.random.normal(mu_dist, sigma_dist)
                    vec_end_short = (vec_end[0] * dist_from_center,
                        vec_end[1] * dist_from_center)
                    start_point = tuple(np.add(center, vec_end_short))
                    start_point = (int(np.round(start_point[0])),
                        int(np.round(start_point[1])))

                    # find end point of spark
                    end_point = (-1, -1)
                    vec_norm = math.sqrt(vec_end[0] ** 2 + vec_end[1] ** 2)
                    if vec_norm > 0 and img_copy[start_point[1], start_point[0]] >= min_spark_br:
                        vec_end = (vec_end[0] / vec_norm, vec_end[1] / vec_norm)
                        spark_length = np.random.normal(mu_length, sigma_length)
                        if spark_length > 0:
                            vec_end = (vec_end[0] * spark_length, vec_end[1] * spark_length)
                            # rotate vector slightly (determined from Gaussian dist.)
                            rotation = np.random.normal(mu_rotation, sigma_rotation)
                            sin_rotation = math.sin(math.radians(rotation))
                            cos_rotation = math.cos(math.radians(rotation))
                            vec_end = (vec_end[0] * cos_rotation - vec_end[1] * sin_rotation,
                                vec_end[0] * sin_rotation + vec_end[1] * cos_rotation)
                            end_point = tuple(np.add(start_point, vec_end))
                            end_point = (int(np.round_(end_point[0])),
                                int(np.round_(end_point[1])))

                            # draw line from start_point to end_point
                            if start_point[0] >= 0 and start_point[0] < image_size[1] and \
                                start_point[1] >= 0 and start_point[1] < image_size[0] and \
                                end_point[0] >= 0 and end_point[0] < image_size[1] and \
                                end_point[1] >= 0 and end_point[1] < image_size[0]:
                                color = (255)
                                thickness = np.inf
                                while (thickness < min_thickness or thickness > max_thickness):
                                    thickness = int(np.round_(np.random.normal(mu_thickness,
                                        sigma_thickness)))
                                spark_map = cv2.line(spark_map, start_point, end_point,
                                    color, thickness)

    spark_map = cv2.blur(spark_map, (3, 3))
    spark_map = cv2.GaussianBlur(spark_map, (7, 7), 0)

    # add sparks to image
    img_copy = np.add(img_copy, spark_map)

    # add Perlin noise to image
    perlin_mask = np.copy(mask)
    perlin_mask = np.add(mask, np.where(mask == 0, perlin_map, 0))
    perlin_mask = np.where(perlin_mask == 255, 0, perlin_mask)
    img_copy = np.add(img_copy, perlin_mask)

    # reformat image
    img_copy = np.where(img_copy > 255, 255, img_copy)
    img_copy = np.where(img_copy < 0, 0, img_copy)
    img_copy = img_copy.astype(np.uint8)
    img_copy = img_copy.reshape(image_size[0], image_size[1], 1)

    # add image to image array
    image_arr = np.append(image_arr, [img_copy], axis = 0)

# save images to .npy file
np.save(dest_file, image_arr)
