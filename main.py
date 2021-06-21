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

img = cv.imread("data/spb.jpg")
img = autoscale(img)

picEdges, contoursMain = processEdges(detectEdges(img))
for i in range(len(contoursMain)):
    contoursMain[i] = np.reshape(contoursMain[i], (contoursMain[i].shape[0], contoursMain[i].shape[2]))

clearingArea = 255 - cv.dilate(picEdges, np.ones((7, 7)))

contoursFilling = fill(img, clearingArea != 0)

contours = contoursFilling + contoursMain

res = np.zeros(img.shape[:2], dtype=np.uint8)
for contour in contours:
    for i in range(contour.shape[0] - 1):
        cv.line(res, (contour[i][0], contour[i][1]), (contour[i + 1][0], contour[i + 1][1]), 255, thickness=1)

res = 255 - res

output = np.concatenate((img, np.stack((res,res,res), axis=2)), axis=1)

cv.imshow("Result", output)

cv.waitKey(0)