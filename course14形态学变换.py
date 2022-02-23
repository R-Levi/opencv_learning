import cv2 as cv
import numpy as np

src = cv.imread("img\\lena.jpg")
#cv.imshow("orign",src)

#腐蚀
def erode_demo(image):
    print(image.shape)
    gray = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
    ret,binary = cv.threshold(gray,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
    #内核,getStructuringElement函数会返回指定形状和尺寸的结构元素。
    #矩形：MORPH_RECT;交叉形：MORPH_CROSS;椭圆形：MORPH_ELLIPSE;
    kernel = cv.getStructuringElement(cv.MORPH_RECT,(3,3))
    print(kernel)
    dst = cv.erode(binary,kernel)
    cv.imshow("erode_image",dst)

#膨胀
def dilate_demo(image):
    print(image.shape)
    gray = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
    ret,binary = cv.threshold(gray,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
    #内核,getStructuringElement函数会返回指定形状和尺寸的结构元素。
    #矩形：MORPH_RECT;交叉形：MORPH_CROSS;椭圆形：MORPH_ELLIPSE;
    kernel = cv.getStructuringElement(cv.MORPH_RECT,(3,3))
    print(kernel)
    dst = cv.dilate(binary,kernel)
    cv.imshow("dilate_image",dst)

#开操作
def open_demo(image):
    print(image.shape)
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
    print(kernel)
    dst = cv.morphologyEx(binary,cv.MORPH_OPEN,kernel)
    cv.imshow("open_image", dst)

#闭操作
def close_demo(image):
    print(image.shape)
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    print(kernel)
    dst = cv.morphologyEx(binary,cv.MORPH_CLOSE,kernel)
    cv.imshow("close_image", dst)


#顶帽
def top_hat_demo(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    dst = cv.morphologyEx(gray,cv.MORPH_TOPHAT,kernel)
    cv.imshow("top_hat_image", dst)

#黑帽
def black_hat_demo(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    dst = cv.morphologyEx(gray,cv.MORPH_BLACKHAT,kernel)
    cv.imshow("black_hat_image", dst)


#erode_demo(src)
#dilate_demo(src)
'''
#对于彩色图像
src1 = cv.imread("img\\lena.jpg")
cv.imshow("src1",src1)
k = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
dst = cv.erode(src1,k)
cv.imshow("erode_image",dst)
'''

img1 = cv.imread("img\\opening.png")
#cv.imshow("orign1",img1)
img2 = cv.imread("img\\closing.png")
#cv.imshow("orign2",img2)

img3 = cv.imread("img\\lena.jpg")
cv.imshow("orign3",img3)
kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
gradient = cv.morphologyEx(img3, cv.MORPH_GRADIENT, kernel)
#cv.imshow("gradient_image",gradient)

em = cv.erode(img3,kernel)
dm = cv.dilate(img3,kernel)
internal = cv.subtract(img3,em)
external = cv.subtract(dm,img3)
cv.imshow("internal",internal)
cv.imshow("external",external)

#open_demo(img1)
#close_demo(img2)

#top_hat_demo(img1)
#black_hat_demo(img2)

cv.waitKey(0)
cv.destroyAllWindows()
