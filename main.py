import numpy as np
import cv2
from time import process_time_ns
from testground import get_image_dirs, get_images_opencv
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
    if side == "top":
        return img[0:int(img.shape[0] * percent), 0:img.shape[1]]
    if side == "bottom":
        return img[img.shape[0] - int(img.shape[0] * percent):img.shape[0], 0:img.shape[1]]
    if side == "left":
        return img[0:img.shape[0], 0:int(img.shape[1] * percent)]
    if side == "right":
        return img[0:img.shape[0], img.shape[1] - int(img.shape[1] * percent):img.shape[1]]
    """if side == "topleftcorner":
        w = int(width * percent)
        h = int(height * percent)
        cv.rectangle(mask, (0, 0), (width, h), 255, thickness=-1)
        cv.rectangle(mask, (0, 0), (w, height), 255, thickness=-1)"""


def autocrop(image):
    y_nonzero, x_nonzero = np.nonzero(image)
    return image[np.min(y_nonzero):np.max(y_nonzero), np.min(x_nonzero):np.max(x_nonzero)]


def paste(canvas, img, offset, first_image_offset):
    image_shape = img.shape
    y0, x0 = int(offset[0]) + first_image_offset[0], int(offset[1]) + first_image_offset[1]
    canvas[y0:y0 + image_shape[0], x0:x0 + image_shape[1]] = img
    return canvas


def stitch(dir, dimension, overlap):
    """
    stitches a grid of images with % overlap. Dimensions is a tuple of (row, columns)
    dir is the folder with the images
    """

    # get list of all images in dir. Read the first image and store the dimension
    images_dir = get_image_dirs(dir)
    img = simplifier(cv2.imread(images_dir[0], 0), "I1")
    image_shape = img.shape
    # create the empty image for the stitched images to be pasted on to
    stitched = np.zeros(shape=[image_shape[0] * dimension[1], image_shape[0] * dimension[0]],
                        dtype=np.uint8)
    mask = np.zeros(stitched.shape[:2], dtype="uint8")
    empty_image = 255 * np.ones(image_shape, dtype="uint8")

    first_image_offset = [100, 100]
    warp_mode = cv2.MOTION_TRANSLATION
    number_of_iterations = 5000;
    termination_eps = 1e-10;
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations, termination_eps)

    # the offset estimated by the microscope
    config_offset_list = [[0] * 2 for _ in range(dimension[1] * dimension[1])]
    with open(dir + '/' + "TileConfiguration.registered.txt") as f:
        contents = f.readlines()
        for i in range(4, len(contents)):
            # lstrip, rstrip, and split isolates the line string in to a pair
            for j, coord in enumerate(contents[i].lstrip("tile_.tif;0123456789 ").lstrip("(").rstrip(")\n").split(",")):
                # float converts the string with a decimal in to a float and round into an int
                config_offset_list[i - 4][j] = round(float(coord) * 0.5)
    print("first image shape: " + str(image_shape))
    print("config_offset_list: " + str(config_offset_list))

    for picture_index in range(len(images_dir)):
        print("picture_index: " + str(picture_index))
        # at the start we paste the first image at (0,0)
        if picture_index == 0:
            stitched = paste(stitched, img, [0, 0], first_image_offset)
            mask = paste(mask, empty_image, [0, 0], first_image_offset)
            offset_list = [[0, 0]]
            continue

        img = simplifier(cv2.imread(images_dir[picture_index], 0), "I1")
        stitch_image_mask = np.zeros(stitched.shape[:2], dtype="uint8")
        stitch_image_mask = paste(stitch_image_mask, empty_image,
                                [config_offset_list[picture_index][1], config_offset_list[picture_index][0]],
                                first_image_offset)
        overlapped_mask = cv2.bitwise_and(mask, stitch_image_mask)

        ROI = autocrop(cv2.bitwise_and(stitched, stitched, mask=overlapped_mask))

        # print("ROI shape: " + str(ROI.shape))
        """if picture_index < dimension[0]:  # if on first row
            img1ROI = getROI(images[picture_index - 1], "right", overlap)
            img2ROI = getROI(images[picture_index], "left", overlap)
        elif picture_index % dimension[0] == 0:  # if on first column of each row
            img1ROI = getROI(images[picture_index - dimension[1]], "bottom", overlap)
            img2ROI = getROI(images[picture_index], "top", overlap)
        else:  # if on >1 column on >1 row
            img1ROI = getROI(images[picture_index - dimension[1]], "bottom", overlap)
            img2ROI = getROI(images[picture_index], "top", overlap)"""

        img_resized = cv2.resize(img, (int(img.shape[1] / 2), int(img.shape[0] / 2)),
                             interpolation=cv2.INTER_AREA)
        ROI_resized = cv2.resize(ROI, (int(ROI.shape[1] / 2), int(ROI.shape[0] / 2)),
                             interpolation=cv2.INTER_AREA)

        cv2.imshow("img", img_resized)
        cv2.imshow("ROI", ROI_resized)
        cv2.waitKey(0)

        warp_matrix = np.eye(2, 3, dtype=np.float32)
        # Run the ECC algorithm. The results are stored in warp_matrix.
        (cc, warp_matrix) = cv2.findTransformECC(ROI, img, warp_matrix, warp_mode, criteria)
        # adjust for the image -> roi image
        warp_matrix[0][2] *= -1
        warp_matrix[1][2] *= -1
        if picture_index < dimension[0]:  # if on first row
            warp_matrix[0][2] += image_shape[1] - int(ROI.shape[1])
        elif picture_index % dimension[0] == 0:  # if on first column of each row
            warp_matrix[1][2] += image_shape[0] - int(ROI.shape[0])
        else:  # if on >1 column on >1 row
            warp_matrix[0][2] += image_shape[1] - int(ROI.shape[1])

        print(warp_matrix)

        # paste current image on to stitched image
        # stitched[int(warp_matrix[1][2]):int(warp_matrix[1][2]) + image_shape[0],
        # int(warp_matrix[0][2]):int(warp_matrix[0][2]) + image_shape[1]] = images[picture_index]
        stitched = paste(stitched, img, [warp_matrix[1][2] + offset_list[picture_index - 1][1], warp_matrix[0][2] + offset_list[picture_index - 1][0]], first_image_offset)
        mask = paste(mask, empty_image, [warp_matrix[1][2] + offset_list[picture_index - 1][1], warp_matrix[0][2] + offset_list[picture_index - 1][0]], first_image_offset)

        # update the last_pos
        offset_list.append([round(warp_matrix[0][2]), round(warp_matrix[1][2])])
        print("offsetList", end=": ")
        print(offset_list)

        stitchedReized = cv2.resize(stitched, (int(stitched.shape[1] / 4), int(stitched.shape[0] / 4)),
                                    interpolation=cv2.INTER_AREA)
        cv2.imshow("Aligned Image 2", stitchedReized)
        cv2.waitKey(0)
    # return final stitched image
    return stitched


if __name__ == '__main__':
    """images = get_image_dirs('C:/Users/jackc/Desktop/test image sample')

    img0 = simplifier(cv2.imread(images[0], 0), "I1")
    img1 = simplifier(cv2.imread(images[1], 0), "I1")

    overlap = 0.35

    t1_start = process_time_ns()

    img0ROI = img0[0:img0.shape[0], img0.shape[1] - int(img0.shape[1] * overlap):img0.shape[1]]
    img1ROI = img1[0:img1.shape[0], 0:int(img1.shape[1] * overlap)]

    print(img0ROI.shape)
    print(img1ROI.shape)

    stitched_dimension = (1100, 3000)
    im2_aligned = np.zeros(shape=[stitched_dimension[0], stitched_dimension[1]], dtype=np.uint8)

    sz = img0.shape

    warp_mode = cv2.MOTION_TRANSLATION
    warp_matrix = np.eye(2, 3, dtype=np.float32)
    number_of_iterations = 5000;
    termination_eps = 1e-10;
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
    cv2.imshow("Aligned Image 2", im2_aligned)
    cv2.waitKey(0)"""
    imageDir = 'E:/phase images/'
    stitchedImage = stitch(imageDir, (4, 4), 0.38)
    stitchedImage = cv2.resize(stitchedImage, (int(stitchedImage.shape[1] / 2), int(stitchedImage.shape[0] / 2)),
                               interpolation=cv2.INTER_AREA)
    cv2.imshow("Aligned Image 2", stitchedImage)
    cv2.waitKey(0)
