import cv2 as cv
import numpy as np

img = cv.imread('img\\lena.jpg')
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
out = cv.Canny(gray,threshold1=50,threshold2=150,L2gradient=True)
res = cv.bitwise_and(img,img,mask=out)
cv.imshow("res",res)

cv.waitKey(0)
cv.destroyAllWindows()