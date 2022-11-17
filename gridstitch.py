import numpy as np
import cv2 as cv
from time import process_time_ns
from testground import get_image_dirs

t1_start = process_time_ns()

# imgaddress = "C:/Users/jackc/PycharmProjects/pano/images/S2.jpg"

# images = get_image_dirs('E:/phase images 2/')
# images = get_image_dirs('C:/Users/jackc/Desktop/reference1/')
# images = get_image_dirs('C:/Users/jackc/Desktop/test image sample')
images = get_image_dirs("E:/phase images/")
stitch_dimension = (1, 2)
MIN_MATCH_COUNT = 5
ROUND_TO_NEAREST_EVEN = 0
overlap = 0.36

# generate an empty image. with the same image type
stitched_dimension = (8000, 8000)
stitched = np.zeros(shape=[stitched_dimension[0], stitched_dimension[1]], dtype=np.uint8)


def autocrop(image):
    y_nonzero, x_nonzero = np.nonzero(image)
    return image[np.min(y_nonzero):np.max(y_nonzero), np.min(x_nonzero):np.max(x_nonzero)]


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
        cv.rectangle(mask, (0, 0), (width, h), 255, thickness=-1)
        return mask
    if side == "bottom":
        h = int(height * percent)
        cv.rectangle(mask, (0, height - h), (width, height), 255, thickness=-1)
        return mask
    if side == "left":
        w = int(width * percent)
        cv.rectangle(mask, (0, 0), (w, height), 255, thickness=-1)
        return mask
    if side == "right":
        w = int(width * percent)
        cv.rectangle(mask, (width - w, 0), (width, height), 255, thickness=-1)
        return mask
    if side == "topleftcorner":
        w = int(width * percent)
        h = int(height * percent)
        cv.rectangle(mask, (0, 0), (width, h), 255, thickness=-1)
        cv.rectangle(mask, (0, 0), (w, height), 255, thickness=-1)
        return mask


def getROIgridmask(stitched_dimension, width, height, startpoint, side, percent):
    # create a mask image filled with zeros, the size of original image
    mask = np.zeros(stitched_dimension, dtype=np.uint8)

    if side == "bottom":
        h = int(height * percent)
        cv.rectangle(mask, (startpoint[0], startpoint[1] + height - h),
                     (startpoint[0] + width, startpoint[1] + height), 255, thickness=-1)
        return mask
    if side == "right":
        w = int(width * percent)
        cv.rectangle(mask, (startpoint[0] + width - w, startpoint[1]),
                     (startpoint[0] + width, startpoint[1] + height), 255, thickness=-1)
        return mask
    if side == "corner":
        """
        @ represents location of start point
        *-----@-----*
        |     |     |
        |   |/|/|/|/|
        *---|/*-----|
        |   |/|     |
        |   |/|     |
        *---|/*-----*

        """
        w = int(width * percent)
        h = int(height * percent)
        # horizontal part
        cv.rectangle(mask, (startpoint[0], startpoint[1] + height - h),
                     (startpoint[0] + width, startpoint[1] + height), 255, thickness=-1)
        # vertical part
        cv.rectangle(mask, (startpoint[0], startpoint[1] + height),
                     (startpoint[0] + w, startpoint[1] + height + (2 * h)), 255, thickness=-1)
        return mask


orb = cv.ORB_create()

# FLANN parameters
FLANN_INDEX_LSH = 6
index_params = dict(algorithm=FLANN_INDEX_LSH,
                    table_number=6,  # 12
                    key_size=12,  # 20
                    multi_probe_level=2)  # 2
search_params = dict(checks=10)
flann = cv.FlannBasedMatcher(index_params, search_params)

detectiontime = []

for picture_index in range(len(images)):

    print("picture_index: " + str(picture_index))
    # at the start we paste the first image at (0,0)
    if picture_index == 0:
        firstpituret1 = process_time_ns()
        img = cv.imread(images[picture_index], 0)
        img = simplifier(img, "I1")
        cv.waitKey()
        stitched[0:img.shape[0], 0:img.shape[1]] = img
        tx, ty = 0, 0
        # last position of stitched picture, used for making the detection mask
        last_pos = [0, 0]
        offset_list = [[0, 0]]
        firstpicturet2 = process_time_ns()
        print("first picture: " + str((firstpicturet2 - firstpituret1) / pow(10, 9)))
        continue

    # read the next image to be detected and stitched
    img = cv.imread(images[picture_index], 0)
    img = simplifier(img, "I1")
    # find the keypoints with ORB
    detectt1 = process_time_ns()
    if picture_index < stitch_dimension[1]:  # if on first row
        mask = getROIgridmask(stitched_dimension, img.shape[1], img.shape[0], offset_list[picture_index - 1], 'right',
                              overlap)
        kp = orb.detect(stitched, mask)
        kp1 = orb.detect(img, getROImask(img.shape[1], img.shape[0], 'left', overlap))
    elif picture_index % stitch_dimension[1] == 0:  # if on first column of each row
        mask = getROIgridmask(stitched_dimension, img.shape[1], img.shape[0],
                              offset_list[picture_index - stitch_dimension[1]], 'bottom', overlap)
        kp = orb.detect(stitched, mask)
        kp1 = orb.detect(img, getROImask(img.shape[1], img.shape[0], 'top', overlap))
    else:  # if on >1 column on >1 row
        mask = getROIgridmask(stitched_dimension, img.shape[1], img.shape[0],
                              offset_list[picture_index - stitch_dimension[1]], 'corner', overlap)
        kp = orb.detect(stitched, mask)
        kp1 = orb.detect(img, getROImask(img.shape[1], img.shape[0], 'topleftcorner', overlap))
    # compute the descriptors with ORB
    kp, des = orb.compute(stitched, kp)
    kp1, des1 = orb.compute(img, kp1)
    detectt2 = process_time_ns()
    detectiontime.append((detectt2 - detectt1) / pow(10, 9))

    # des = np.float32(des)
    # des1 = np.float32(des1)
    if picture_index < 2:
        print(kp)
        print(des)

    matches = flann.knnMatch(des, des1, k=2)

    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)

    """showMatches = cv.drawMatches(stitched, kp, img, kp1, good, None)
    resized1 = cv.resize(showMatches, (int(showMatches.shape[0] / 4), int(showMatches.shape[1] / 4)),
                        interpolation=cv.INTER_AREA)
    cv.imshow("correspondences", resized1)
    cv.waitKey()"""

    """resized2 = cv.resize(mask, (int(mask.shape[0] / 4), int(mask.shape[1] / 4)), interpolation=cv.INTER_AREA)
    resized1 = cv.resize(stitched, (int(stitched.shape[0] / 4), int(stitched.shape[1] / 4)),
                         interpolation=cv.INTER_AREA)
    dst = cv.addWeighted(resized1, 1, resized2, 0.3, 0)
    cv.imshow('Blended Image', dst)
    cv.waitKey()"""

    if len(good) > MIN_MATCH_COUNT:
        # -- Localize the object
        src_pts = np.empty((len(good), 2), dtype=np.float32)
        dst_pts = np.empty((len(good), 2), dtype=np.float32)
        for i in range(len(good)):
            # -- Get the keypoints from the good matches
            src_pts[i, 0] = kp[good[i].queryIdx].pt[0]
            src_pts[i, 1] = kp[good[i].queryIdx].pt[1]
            dst_pts[i, 0] = kp1[good[i].trainIdx].pt[0]
            dst_pts[i, 1] = kp1[good[i].trainIdx].pt[1]
        # mask - Nx1 mask array of same length as input points, indicates inliers (which points were actually used in the
        # best computation of H).

        # the for loop below calculates only the translational homography instead of using the homography function
        # truncate_amount specifies how much of each end of the list to remove
        truncate_amount = 2
        dif_tx = []
        dif_ty = []

        for src, dst in zip(src_pts, dst_pts):
            dif_tx.append(src[0] - dst[0])
            dif_ty.append(src[1] - dst[1])
        dif_tx.sort()
        dif_ty.sort()
        tx = sum(dif_tx[truncate_amount:-truncate_amount]) / (len(src_pts) - truncate_amount * 2)
        ty = sum(dif_ty[truncate_amount:-truncate_amount]) / (len(src_pts) - truncate_amount * 2)

    else:
        print("Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT))
        matchesMask = None

    # paste the two images using the homography matrix for translation
    if ROUND_TO_NEAREST_EVEN:
        tx /= 2
        ty /= 2
        tx = (round(tx)) * 2
        ty = (round(ty)) * 2
    stitched[int(ty):int(ty) + img.shape[0], int(tx):int(tx) + img.shape[1]] = img

    # update the last_pos
    offset_list.append([round(tx), round(ty)])
    print(tx, end=', ')
    print(ty)

t1_stop = process_time_ns()

print("Elapsed time:", t1_stop / pow(10, 9), t1_start / pow(10, 9))

print("Elapsed time during the whole program in seconds:",
      (t1_stop - t1_start) / pow(10, 9))

print("per image:",
      (t1_stop - t1_start) / (pow(10, 9) * 16))

stitchedresized = cv.resize(stitched, (int(stitched.shape[0] / 8), int(stitched.shape[1] / 8)),
                            interpolation=cv.INTER_AREA)

cv.imshow("stitched", autocrop(stitchedresized))
cv.waitKey()

print(sum(detectiontime) / len(detectiontime))
