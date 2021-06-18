import cv2 as cv
import numpy as np
from src.detectEdges import detectEdges, processEdges
from src.filling import fill

def autoscale(img):
    w, h = img.shape[:2]
    if w > h:
        wn = 512
        hn = 512 * (h / w)
    else:
        hn = 512
        wn = 512 * (w / h)
    return cv.resize(img, (int(hn), int(wn)))

img = cv.imread("src/lena_std.tif")
img = autoscale(img)

pic, contours = processEdges(detectEdges(img))

picFilling = fill(img)

res = 255 - cv.bitwise_or(pic, picFilling)

output = np.concatenate((img, np.stack((res,res,res), axis=2)), axis=1)

cv.imshow("Result", output)

cv.waitKey(0)