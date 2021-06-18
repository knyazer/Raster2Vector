# import the necessary packages
from skimage.segmentation import slic
from skimage import io
import numpy as np
import cv2 as cv

def fill(image):
    image = np.array(image, dtype=np.float32) / 255
    segments = slic(image, n_segments=400, sigma=5)

    N = np.max(segments)

    output = np.zeros(image.shape[:2], dtype=np.uint8)
    hatched = np.zeros(image.shape[:2], dtype=np.uint8)

    for i in range(N):
        mask = (segments == i)
        mean = 255 * np.array(cv.mean(image, mask=np.array(mask, dtype=np.uint8) * 255)[:3])
        mean = np.reshape(np.array(mean, dtype=np.uint8), (1,1,3))
        mean = cv.cvtColor(mean, cv.COLOR_BGR2LAB)[0,0,0]
        mean = 255 - mean

        step = 3 / (float(mean) / 255) + 1.5
        pattern = np.zeros(image.shape[:2], dtype=np.uint8)
        for p in range(0, pattern.shape[0], int(step)):
            cv.line(pattern, (0, p), (pattern.shape[1], p), 255, thickness=1)

        output += np.array(int(mean * 255) * mask, dtype=np.uint8)
        hatched += pattern * (mask != 0)
    
    return hatched