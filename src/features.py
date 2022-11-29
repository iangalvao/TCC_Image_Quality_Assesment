from matplotlib import colors
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
from matplotlib.pyplot import imshow
from skimage.color import rgb2lab, lab2rgb
import scipy.ndimage.filters


def hist_width(v, p):
    threshold = (1 - p) / 2
    leftsum = 0
    rightsum = 0
    l = len(v)
    end = l - 1
    start = 0
    left = 1
    right = 1
    for i in range(l):
        leftsum += v[i]
        rightsum += v[l - 1 - i]
        if (leftsum >= threshold) and left:
            start = i
            left = 0
        if (rightsum >= threshold) and right:
            end = l - 1 - i
            right = 0
        if (not left) and (not right):
            break
    if start > end:
        result = 0
    else:
        result = (end - start) / l
    return result


def min_range(v, p):
    w = len(v)
    sumByPos = np.zeros(w)
    resRange = []
    for i in range(w):
        resRange.append(w + 1)
        for j in range(i, w):
            sumByPos[i] += v[j]
            if sumByPos[i] > p:
                resRange[i] = j - i
                break
    x = min(resRange)

    return x


def hue_simplicity(imArray):
    hsvimage = colors.rgb_to_hsv(imArray)
    hue = hsvimage[:, :, 0]
    sat = hsvimage[:, :, 1]
    value = hsvimage[:, :, 2]

    normV = value / 255
    saturated = [(p > 0.2) for p in sat.flatten()]
    middleValues = [((p > 0.15) and (p < 0.95)) for p in normV.flatten()]
    filteredHues = hue.flatten()[saturated and middleValues]

    hist = np.histogram(filteredHues, bins=20)
    maxHue = max(hist[0])
    t = maxHue * 0.05
    count = 0
    for i in range(20):
        n = hist[0][i]
        if n > t:
            count += 1
    feature = 20 - count
    return feature


def luminanceContrast(imArray):

    labimage = rgb2lab(imArray)

    lum = labimage[:, :, 0]
    a = labimage[:, :, 1]
    b = labimage[:, :, 2]

    fLum = lum.flatten()

    maxL = max(fLum)

    minL = min(fLum)

    averageL = np.average(fLum)
    ratio = minL / maxL

    r = imArray[:, :, 0]
    g = imArray[:, :, 1]
    b = imArray[:, :, 2]
    histR = np.histogram(r, bins=range(256))
    histG = np.histogram(g, bins=range(256))
    histB = np.histogram(b, bins=range(256))

    sumHist = histR[0] + histG[0] + histB[0]

    histWidth = hist_width(sumHist / sumHist.sum(), 0.98)

    f = histWidth / 256
    return (averageL, ratio, f)


""" 
def edge_simplicity(imArray):
    laplacian = scipy.ndimage.filters.laplace(imArray[:, :, 0])

    totalSum = laplacian.sum()
    sumRows = laplacian.sum(axis=1) / totalSum
    sumColumns = laplacian.sum(axis=0) / totalSum

    f1 = hist_width(sumColumns, 0.95) / len(sumRows)
    f2 = hist_width(sumRows, 0.95) / len(sumColumns)
    return (f1 + f2) / 2
 """


def edge_simplicity(imArray):
    blur = scipy.ndimage.gaussian_filter(imArray, 2)
    laplacian = scipy.ndimage.laplace(blur)

    totalSum = laplacian.sum()
    sumRows = laplacian.sum(axis=1) / totalSum
    sumRows = sumRows.sum(axis=1)
    sumColumns = laplacian.sum(axis=0) / totalSum
    sumColumns = sumColumns.sum(axis=1)
    f1 = hist_width(sumColumns, 0.98) / len(sumRows)
    f2 = hist_width(sumRows, 0.98) / len(sumColumns)
    return (f1 + f2) / 2


def blur(imArray):
    fft = np.fft.fft2(imArray[:, :, 0]).flatten()
    alpha = 5
    filtro = [(a > alpha) for a in fft]
    C = fft[filtro]
    feat = len(C) / len(fft)
    return feat


def extractFeatures(imArray):
    e = edge_simplicity(imArray)
    # h = hue_simplicity(imArray)
    # al, r, hw = luminanceContrast(imArray)
    # b = blur(imArray)
    return [e]  # , h, al, r, hw, b]
