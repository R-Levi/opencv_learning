import cv2 as cv
import numpy as np

def fill_color_demo(image):
    copyImg = image.copy()
    h,w = image.shape[:2]
    mask = np.zeros([h+2,w+2],np.uint8)
    #mask[30:150,30:150] = 0
    #cv.imshow('mask',mask)
    cv.floodFill(copyImg,mask=mask,seedPoint=(25,130),
                 newVal=(174,199,227),loDiff=(100,100,100),upDiff=(50,50,50),flags=cv.FLOODFILL_FIXED_RANGE)
    cv.imshow('fill_color',copyImg)

def fill_binary():
    image = np.zeros([400,400,3],np.uint8)
    image[100:300,100:300] = 255
    cv.imshow('img',image)

    mask = np.ones([402,402],np.uint8)
    mask[101:301,101:301]=0
    cv.floodFill(image,mask,(200,200),(255,255,0),cv.FLOODFILL_MASK_ONLY)
    cv.imshow('fill_binary',image)

src = cv.imread('img\gxt.jpg')
print(src.shape)
cv.imshow('test',src)
fill_color_demo(src)
#fill_binary()
'''
face = src[25:180,130:255]
gray = cv.cvtColor(face,cv.COLOR_BGR2GRAY)
backface = cv.cvtColor(gray,cv.COLOR_GRAY2BGR)
src[25:180,130:255] = backface
cv.imshow('face',src)
'''

cv.waitKey(0)
cv.destroyAllWindows()