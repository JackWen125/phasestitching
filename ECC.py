import numpy as np
import matplotlib.pyplot as plt
import cv2
from testground import get_image_dirs
from time import process_time_ns


def simplifier(im, p):
    # im = cv.imread(f, 0)
    columns = len(im[0])
    rows = len(im)
    if p == "I3":
        I3 = (im[0:rows:2, 0:columns:2]).astype(np.uint8)  # this is I3, it works
        return I3
    elif p == "I2":
        I2 = (im[0:rows:2, 1:columns:2]).astype(np.uint8)  # this is I2, it works
        return I2
    elif p == "I1":
        I1 = (im[1:rows:2, 1:columns:2]).astype(np.uint8)  # this is I1, it works
        return I1
    elif p == "I4":
        I4 = (im[1:rows:2, 0:columns:2]).astype(np.uint8)  # this is I4, it works
        return I4
    else:
        print("invalid input for p")


def getROImask(width, height, side, percent):
    # create a mask image filled with zeros, the size of original image
    mask = np.zeros((width, height), dtype=np.uint8)

    if side == "top":
        h = int(height * percent)
        # draw your selected ROI on the mask image
        cv2.rectangle(mask, (0, 0), (width, h), 255, thickness=-1)
        return mask
    if side == "bottom":
        h = int(height * percent)
        cv2.rectangle(mask, (0, height - h), (width, height), 255, thickness=-1)
        return mask
    if side == "left":
        w = int(width * percent)
        cv2.rectangle(mask, (0, 0), (w, height), 255, thickness=-1)
        return mask
    if side == "right":
        w = int(width * percent)
        cv2.rectangle(mask, (width - w, 0), (width, height), 255, thickness=-1)
        return mask
    if side == "topleftcorner":
        w = int(width * percent)
        h = int(height * percent)
        cv2.rectangle(mask, (0, 0), (width, h), 255, thickness=-1)
        cv2.rectangle(mask, (0, 0), (w, height), 255, thickness=-1)
        return mask

def getROI(img, side, percent):
    # returns part of image specified by side and percent
    if side == "top":
        return img[int(img.shape[0] * percent):img.shape[0], 0:img.shape[1]]
    if side == "bottom":
        return img[0:img.shape[0] - int(img.shape[0] * percent), 0:img.shape[1]]
    if side == "left":
        return img[0:img.shape[0], 0:int(img.shape[1] * percent)]
    if side == "right":
        return img[0:img.shape[0], img.shape[1] - int(img.shape[1] * percent):img.shape[1]]


if __name__ == '__main__':
    # images = get_image_dirs('C:/Users/jackc/Desktop/test image sample')
    # images = get_image_dirs('E:/phase images 2/')
    images = get_image_dirs('C:/Users/jackc/Desktop/reference1/')

    img0 = simplifier(cv2.imread(images[0], 0), "I1")
    img1 = simplifier(cv2.imread(images[1], 0), "I1")

    overlap = 0.35


    t1_start = process_time_ns()

    """img0ROI = img0[0:img0.shape[0], img0.shape[1] - int(img0.shape[1] * overlap):img0.shape[1]]
    img1ROI = img1[0:img1.shape[0], 0:int(img1.shape[1] * overlap)]"""

    img0ROI = getROI(img0, 'right', overlap)
    img1ROI = img1

    print(img0ROI.shape)
    print(img1ROI.shape)

    stitched_dimension = (1100, 3000)
    im2_aligned = np.zeros(shape=[stitched_dimension[0], stitched_dimension[1]], dtype=np.uint8)

    sz = img0.shape

    warp_mode = cv2.MOTION_TRANSLATION
    warp_matrix = np.eye(2, 3, dtype=np.float32)
    number_of_iterations = 5000;
    termination_eps = 1e-6;
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations, termination_eps)

    # Run the ECC algorithm. The results are stored in warp_matrix.
    (cc, warp_matrix) = cv2.findTransformECC(img0ROI, img1ROI, warp_matrix, warp_mode, criteria)
    # adjust for the image -> roi image
    warp_matrix[0][2] *= -1
    warp_matrix[0][2] += img1.shape[1] - int(img1.shape[1] * overlap)
    warp_matrix[1][2] *= -1
    print(warp_matrix)

    t1_stop = process_time_ns()
    print("Elapsed time during the whole program in seconds:",
          (t1_stop - t1_start) / pow(10, 9))

    # Use warpAffine for Translation, Euclidean and Affine
    im2_aligned[0:img0.shape[0], 0:img0.shape[1]] = img0
    im2_aligned[int(warp_matrix[1][2]):int(warp_matrix[1][2]) + img1.shape[0],
    int(warp_matrix[0][2]):int(warp_matrix[0][2]) + img1.shape[1]] = img1
    # im2_aligned = cv2.warpAffine(img0, warp_matrix, (sz[1], sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP);
    # Show final results
    img0 = cv2.resize(img0, (int(img0.shape[1] / 2), int(img0.shape[0] / 2)),
                      interpolation=cv2.INTER_AREA)
    img1 = cv2.resize(img1, (int(img1.shape[1] / 2), int(img1.shape[0] / 2)),
                      interpolation=cv2.INTER_AREA)
    im2_aligned = cv2.resize(im2_aligned, (int(im2_aligned.shape[1] / 2), int(im2_aligned.shape[0] / 2)),
                             interpolation=cv2.INTER_AREA)
    # cv2.imshow("Image 1", img0)
    # cv2.imshow("Image 2", img1)
    """cv2.imshow("Aligned Image 2", im2_aligned)
    cv2.waitKey(0)"""
    plt.imshow(im2_aligned, cmap='gray')
    plt.show()
