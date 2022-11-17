# sum(abs((reference intensity - average reference intensity) - (gtruth intensity - average gtruth intensity)))
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity
from testground import get_image_dirs
import cv2 as cv


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


# referenceImageDir = get_image_dirs('S:/Summer 2022 Zhesi Affat/100622/reference1/reference1_1/split/')
# gTruthImageDir = get_image_dirs('S:/Summer 2022 Zhesi Affat/100622/gtruth1/gtruth1_1/split/')

referenceImageDir = get_image_dirs('C:/Users/jackc/Desktop/reference1/')

referenceImages = [simplifier(cv.imread(referenceImageDir[0], 0), "I1"),
                   simplifier(cv.imread(referenceImageDir[1], 0), "I1")]


# gTruthImage = Image.open(simplifier(cv.imread(gTruthImageDir[0], 0), "I1"))

# 2448 x 2048
# 1224 x 1024
# 612 x 512

def getSSD(img1, img2, offset):
    # offset[ y (rows), x (cols)]
    overlap1 = img1[round(offset[0]):1024, 1224 - round(offset[1]):1224]
    overlap2 = img2[0:1024 - round(offset[0]), 0:round(offset[1])]

    """print(overlap1.shape)
    print(overlap2.shape)
    cv.imshow("Image 1", img1)
    cv.imshow("Image 2", img2)
    cv.imshow("overlap 1", overlap1)
    cv.imshow("overlap 2", overlap2)
    cv.waitKey(0)"""

    ssd = np.sum((overlap1.astype("float") - overlap2.astype("float")) ** 2)
    ssd /= float(overlap1.shape[0] * overlap1.shape[1])

    return ssd

def getSSIM(img1, img2, offset):
    # offset[ y (rows), x (cols)]
    overlap1 = img1[round(offset[0]):1024, 1224 - round(offset[1]):1224]
    overlap2 = img2[0:1024 - round(offset[0]), 0:round(offset[1])]
    # Compute SSIM between the two images
    (score, diff) = structural_similarity(overlap1, overlap2, full=True)
    # print("Image Similarity: {:.4f}%".format(score * 100))
    return [score, diff]


if __name__ == '__main__':
    fijiOffset = [[818.2159424, 0.5877462626]
        , [818.4682584, 0.4682583511]
        , [818.2671509, 0.5224394798]
        , [818.1174927, 0.4673044596]
        , [818.43573, 0.6798152328]
        , [818.2866211, 0.6654381156]
        , [818.2224121, 0.638455689]]

    eccOffset = [[818.11206, 1.1256537]
        , [818.1205, 1.1376417]
        , [818.105, 1.1472152]
        , [818.1223, 1.1537757]
        , [818.1124, 1.1530544]
        , [818.09515, 1.1556525]
        , [818.10034, 1.1530241]
        , [818.1249, 1.1476943]
        , [818.0968, 1.138541]
        , [818.0973, 1.1464351]]

    orbOffset = [[818.1109245818795, 1.303871525144114]
        , [818.1720848446801, 1.3075619652157737]
        , [818.1745083148663, 1.3062884990985577]
        , [818.1805759645858, 1.314788818359375]
        , [818.1941001962084, 1.2997391726992547]
        , [818.1941001962084, 1.2997391726992547]
        , [818.2623750232515, 1.2073407854352678]
        , [818.2623750232515, 1.2073407854352678]
        , [818.2837717484455, 1.3496012784996811]
        , [818.1665985107422, 1.233397216796875]
        , [818.0957193180006, 1.3332494618941326]
        , [818.1565920511881, 1.2566852569580078]]

    fijiSSD = []
    for offset in fijiOffset:
        fijiSSD.append(getSSD(referenceImages[0], referenceImages[1], [offset[1] * 0.5, offset[0] * 0.5]))

    eccSSD = []
    for offset in eccOffset:
        eccSSD.append(getSSD(referenceImages[0], referenceImages[1], [offset[1] * 0.5, offset[0] * 0.5]))

    orbSSD = []
    for offset in orbOffset:
        orbSSD.append(getSSD(referenceImages[0], referenceImages[1], [offset[1] * 0.5, offset[0] * 0.5]))

    print('average fiji ssd: ', end='')
    print(sum(fijiSSD)/len(fijiSSD))
    print('std fiji ssd: ', end='')
    print(np.std(fijiSSD))
    print('average ecc ssd: ', end='')
    print(sum(eccSSD)/len(eccSSD))
    print('std ecc ssd: ', end='')
    print(np.std(eccSSD))
    print('average orb ssd: ', end='')
    print(sum(orbSSD) / len(orbSSD))
    print('std orb ssd: ', end='')
    print(np.std(orbSSD))

    fijiSSIM = []
    for offset in fijiOffset:
        fijiSSIM.append(getSSIM(referenceImages[0], referenceImages[1], [offset[1] * 0.5, offset[0] * 0.5]))
    eccSSIM = []
    for offset in eccOffset:
        eccSSIM.append(getSSIM(referenceImages[0], referenceImages[1], [offset[1] * 0.5, offset[0] * 0.5]))
    orbSSIM = []
    for offset in orbOffset:
        orbSSIM.append(getSSIM(referenceImages[0], referenceImages[1], [offset[1] * 0.5, offset[0] * 0.5]))

    print("average fiji Similarity: {:.4f}%".format((sum(i[0] for i in fijiSSIM) / len(fijiSSD)) * 100))
    # print('average fiji SSIM diff: ', end='')
    # print(sum(i[1] for i in fijiSSIM) / len(fijiSSD))
    print("average ecc Similarity: {:.4f}%".format((sum(i[0] for i in eccSSIM) / len(eccSSIM)) * 100))
    # print('average ecc SSIM diff: ', end='')
    # print(sum(i[1] for i in eccSSIM) / len(eccSSIM))
    print("average orb Similarity: {:.4f}%".format((sum(i[0] for i in orbSSIM) / len(orbSSIM)) * 100))

    # https://stackoverflow.com/questions/56183201/detect-and-visualize-differences-between-two-images-with-opencv-python