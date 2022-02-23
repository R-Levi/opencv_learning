import cv2 as cv
import numpy as np

def color_sapce(image):
    gray = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
    cv.imshow('gray',gray)

    hsv = cv.cvtColor(image,cv.COLOR_BGR2HSV)
    cv.imshow('hsv',hsv)

    yuv = cv.cvtColor(image,cv.COLOR_BGR2YUV)
    cv.imshow('yuv',yuv)

    #cv.waitKey(0)


def extract_object():
    capture = cv.VideoCapture('img\\flower.mp4')
    while(True):
        ret,frame = capture.read()
        if not ret:
            break
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        #色彩提取追踪
        lower_hsv = np.array([26,43,46])
        higher_hsv = np.array([34,255,255])
        mask = cv.inRange(hsv,lowerb=lower_hsv,upperb=higher_hsv)
        dst = cv.bitwise_and(frame,frame,mask=mask)
        cv.imshow('video',frame)
        cv.imshow('mask',dst)
        c = cv.waitKey(40)
        if c==27:#ESC的ASCII
            break



'''
src = cv.imread('img\dog.jpg')
b,g,r = cv.split(src)
cv.imshow('b',b)
cv.imshow('g',g)
cv.imshow('r',r)
cv.waitKey(0)
cv.destroyAllWindows()
src = cv.merge([b,g,r],src)
src[:,:,2] = 0
cv.imshow('merge',src)
cv.waitKey(0)
'''
t1 = cv.getTickCount()
extract_object()
t2 = cv.getTickCount()
print((t2-t1)/cv.getTickFrequency())