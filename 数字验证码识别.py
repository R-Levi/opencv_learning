import cv2 as cv
import numpy as np
import pytesseract as tess
from PIL import Image

def recognize_test(image):
    gray = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
    ret,binary = cv.threshold(gray,0,255,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
    k1 = cv.getStructuringElement(cv.MORPH_RECT,(1,6))
    bin1 = cv.morphologyEx(binary, cv.MORPH_OPEN, k1)
    k2 = cv.getStructuringElement(cv.MORPH_RECT,(5,1))
    bin2 = cv.morphologyEx(bin1,cv.MORPH_OPEN,k2)
    cv.imshow("bin",bin2)

    textImage1 = np.asarray(bin2)
    textImage = Image.fromarray(bin2)
    print(type(textImage))
    print(type(textImage1))
    text = tess.image_to_string(textImage1)
    print("验证码:%s"%text)



src = cv.imread('img\\yzm2.png')
cv.imshow('image',src)
recognize_test(src)

cv.waitKey(0)
cv.destroyAllWindows()
