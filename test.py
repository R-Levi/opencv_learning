import cv2 as cv
import numpy as np

src = cv.imread('img\\ghost.jpg')
cv.imshow('image',src)
cv.waitKey(0)
cv.destroyAllWindows()
print("OPENCV")