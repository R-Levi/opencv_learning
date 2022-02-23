import cv2 as cv
import numpy as np

def add_demo(m1,m2):
    dst = cv.add(m1,m2)
    cv.imshow('add_demo',dst)

def sub_demo(m1,m2):
    dst = cv.subtract(m2,m1)
    cv.imshow('sub_demo',dst)


def divid_demo(m1,m2):
    dst = cv.divide(m1,m2)
    cv.imshow('divid',dst)

def multiply_demo(m1,m2):
    dst = cv.multiply(m1,m2)
    cv.imshow('mutiply',dst)

def others(m1,m2):
    #均值 方差
    h1,dev1 = cv.meanStdDev(m1)
    h2,dev2= cv.meanStdDev(m2)

    print(h1)
    print(h2)

    print(dev1)
    print(dev2)

#逻辑运算
def logic_demo(m1,m2):
    dst1 = cv.bitwise_and(m1,m2)
    dst2 = cv.bitwise_or(m1,m2)
    dst3 = cv.bitwise_not(m1)
    cv.imshow('and',dst1)
    cv.imshow('or',dst2)
    cv.imshow('not',dst3)

#调整亮度对比度
def control_bright_demo(image,c,b):
    h,w,channels = image.shape
    blank = np.zeros([h,w,channels],image.dtype)
    cv.imshow('blank',blank)
    dst = cv.addWeighted(image,c,blank,1-c,b)
    cv.imshow('dst',dst)

src1 = cv.imread('img\WindowsLogo.jpg')
src2 = cv.imread('img\LinuxLogo.jpg')

src = cv.imread('img\dog.jpg')
cv.imshow('dog',src)
#control_bright_demo(src,1.2,10)
#add_demo(src1,src2)
#sub_demo(src1,src2)
#divid_demo(src1,src2)
#multiply_demo(src1,src2)
#others(src1,src2)
#cv.imshow('orgin',src1)
logic_demo(src1,src2)
'''
#色彩跟踪
src = cv.imread('img\dog.jpg')
cv.imshow('orign_dog',src)
hsv = cv.cvtColor(src,cv.COLOR_BGR2HSV)
low = np.array([35,43,46])
up = np.array([77,255,255])
mask = cv.inRange(hsv,lowerb=low,upperb=up)
cv.imshow('mask',mask)
dst = cv.bitwise_and(src,src,mask=mask)
cv.imshow('dst',dst)
'''
cv.waitKey(0)
cv.destroyAllWindows()