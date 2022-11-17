from PIL import Image
import numpy as np
import cv2 as cv
from testground import get_image_dirs
from matplotlib import pyplot as plt

# imgaddress = "C:/Users/jackc/PycharmProjects/pano/images/S2.jpg"

# images = get_image_dirs('E:/phase images/')
images = get_image_dirs('C:/Users/jackc/Desktop/test image sample')
MIN_MATCH_COUNT = 5
picture_index = [0, 1]

# 1 or 0, on or off
ROUND_TO_NEAREST_EVEN = 1


def getROImask(image, side, percent):
    x = image.shape[0]
    y = image.shape[1]
    # create a mask image filled with zeros, the size of original image
    mask = np.zeros(image.shape[:2], dtype=np.uint8)

    if side == "top":
        h = int(y * percent)
        # draw your selected ROI on the mask image
        cv.rectangle(mask, (0, 0), (x, h), 255, thickness=-1)
        return mask
    if side == "bottom":
        h = int(y * percent)
        cv.rectangle(mask, (0, y - h), (x, y), 255, thickness=-1)
        return mask
    if side == "left":
        w = int(x * percent)
        cv.rectangle(mask, (0, 0), (w, y), 255, thickness=-1)
        return mask
    if side == "right":
        w = int(x * percent)
        cv.rectangle(mask, (x - w, 0), (x, y), 255, thickness=-1)
        return mask


img = cv.imread(images[picture_index[0]], 0)
img1 = cv.imread(images[picture_index[1]], 0)
# Initiate ORB detector
orb = cv.ORB_create()
# find the keypoints with ORB
kp = orb.detect(img, getROImask(img, 'right', 0.21))
kp1 = orb.detect(img1, getROImask(img1, 'left', 0.21))
# compute the descriptors with ORB
kp, des = orb.compute(img, kp)
kp1, des1 = orb.compute(img1, kp1)

# draw only keypoints location,not size and orientation
# img2 = cv.drawKeypoints(img, kp, None, color=(0,255,0), flags=0)
# plt.imshow(img2), plt.show()
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm=0, trees=5)
search_params = dict(checks=50)
flann = cv.FlannBasedMatcher(index_params, search_params)

des = np.float32(des)
des1 = np.float32(des1)

matches = flann.knnMatch(des, des1, k=2)

good = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good.append(m)

"""
img3 = cv.drawMatches(img, kp, img1, kp1, good, None)
resized = cv.resize(img3, (int(img3.shape[0] / 2), int(img3.shape[1] / 8)), interpolation=cv.INTER_AREA)
cv.imshow("correspondences", resized)
cv.waitKey()
"""
# H, __ = cv.findHomography(srcPoints, dstPoints, cv.RANSAC, 4)
# H, __ = cv.findHomography(des, des1, cv.RANSAC, 4)

if len(good) > MIN_MATCH_COUNT:
    """
    src_pts = np.float32([kp[m.queryIdx].pt for m in good]).reshape(-1, 2)
    dst_pts = np.float32([kp[m.trainIdx].pt for m in good]).reshape(-1, 2)
    """
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
        dif_tx.append(dst[0] - src[0])
        dif_ty.append(dst[1] - src[1])
    dif_tx.sort()
    dif_ty.sort()
    tx = sum(dif_tx[truncate_amount:-truncate_amount]) / (len(src_pts) - truncate_amount * 2)
    ty = sum(dif_ty[truncate_amount:-truncate_amount]) / (len(src_pts) - truncate_amount * 2)

    H = np.zeros((3, 3))
    H[0][2] = tx
    H[1][2] = ty
    # H, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 4.0, maxIters=2000, confidence=0.999)
    """
    H[0][0] = 1.0
    H[0][1] = 0.0
    H[1][0] = 0.0
    H[1][1] = 1.0
    H[2][0] = 0.0
    H[2][1] = 0.0
    """
    # print(H)

    # matchesMask = mask.ravel().tolist()
    """
    h, w = img1.shape
    pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
    dst = cv.perspectiveTransform(pts, H)
    """

else:
    print("Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT))
    matchesMask = None

# plt.imshow(img4, 'gray'), plt.show()

"""
draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                   singlePointColor=None,
                   matchesMask=matchesMask,  # draw only inliers
                   flags=2)
img3 = cv.drawMatches(img, kp, img1, kp1, good, None, **draw_params)
plt.imshow(img3, 'gray'), plt.show()
"""

"""
xh = np.linalg.inv(H)
f1 = np.dot(xh, np.array([0, 0, 1]))
f1 = f1/f1[-1]
xh[0][-1] += abs(f1[0])
xh[1][-1] += abs(f1[1])
ds = np.dot(xh, np.array([img1.shape[1], img1.shape[0], 1]))
offsety = abs(int(f1[1]))
offsetx = abs(int(f1[0]))
dsize = (int(ds[0])+offsetx, int(ds[1]) + offsety)
# tmp = cv.warpPerspective(img1, xh, dsize)
"""


# generate an empty image. with the same image type
# img_x = (Image.open(images[0]).size[0])
# img_y = (Image.open(images[0]).size[1])
img_mode = Image.open(images[0]).mode
new_image = Image.new(img_mode, (8000, 4000))

# paste the two images using the homggraphy matrix for translation
pilimg = Image.open(images[picture_index[0]])
pilimg1 = Image.open(images[picture_index[1]])
# new_image.paste(pilimg1, (pilimg1.size[0], 0))
# new_image.paste(pilimg, (int(H[0][2]) + pilimg1.size[0], int(H[1][2])))
new_image.paste(pilimg, (0, 0))
if ROUND_TO_NEAREST_EVEN:
    H[0][2] /= 2
    H[1][2] /= 2
    H[0][2] = (round(H[0][2])) * 2
    H[1][2] = (round(H[1][2])) * 2
new_image.paste(pilimg1, (-round(H[0][2]), -round(H[1][2])))

new_image.show()
# resized = cv.resize(new_image, (int(new_image.shape[0] / 2), int(new_image.shape[1] / 8)), interpolation=cv.INTER_AREA)
# cv.imshow("warped", new_image)
# cv.waitKey()
