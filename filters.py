import skimage.filters
import cv2
import numpy as np
from testground import get_image_dirs

images = get_image_dirs('E:/phase images/')
image = cv2.imread(images[0], 0)
# 3x3 sobel filter for horizontal edge detection
sobel_y = np.array([[ -1, -2, -1],
                   [ 0, 0, 0],
                   [ 1, 2, 1]])
# vertical edge detection
sobel_x = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]])
# filter the image using filter2D(grayscale image, bit-depth, kernel)
filtered_image1 = cv2.filter2D(image, -1, sobel_y)
filtered_image2 = cv2.filter2D(image, -1, sobel_x)
"""cv2.imshow('filter 1', filtered_image1)
cv2.waitKey()
cv2.imshow('filter 2', filtered_image2)
cv2.waitKey()"""
blended = cv2.addWeighted(filtered_image1, 0.5, filtered_image2, 0.5, 0)


def gaussian_kernel_generator(size, sigma=1):
    size = int(size) // 2
    x, y = np.mgrid[-size:size+1, -size:size+1]
    normal = 1 / (2.0 * np.pi * sigma**2)
    g = np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normal
    return g

highpass_kernel = np.array([[0.0, -1.0, 0.0],
                   [-1.0, 4.0, -1.0],
                   [0.0, -1.0, 0.0]])

highpass_kernel = highpass_kernel/(np.sum(highpass_kernel) if np.sum(highpass_kernel)!=0 else 1)

# blur
# gaussian_kernel = gaussian_kernel_generator(3)
blended = skimage.filters.gaussian(
    image, sigma=(2, 2), truncate=3.5)

# filter the source image
img_rst = cv2.filter2D(blended, -1, highpass_kernel)

resized = cv2.resize(img_rst, (int(img_rst.shape[1] / 2), int(img_rst.shape[0] / 2)), interpolation=cv2.INTER_NEAREST)
cv2.imshow('filter 2', resized)
cv2.waitKey()