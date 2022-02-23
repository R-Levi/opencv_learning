import numpy as np
import cv2 as cv

src = cv.imread('img\\sudoku.png')

def sobel_demo(image):
    #x,y梯度
    grad_x = cv.Sobel(image,cv.CV_32F,1,0)
    grad_y = cv.Sobel(image,cv.CV_32F,0,1)
    #取绝对值
    gradx = cv.convertScaleAbs(grad_x)
    grady = cv.convertScaleAbs(grad_y)

    cv.imshow("grad_x",gradx)
    cv.imshow("grad_y",grady)

    gradxy = cv.addWeighted(gradx,0.5,grady,0.5,0)
    cv.imshow("result",gradxy)
def lpls_demo(image):
    dst = cv.Laplacian(image,cv.CV_32F,ksize=3)
    lpls = cv.convertScaleAbs(dst)
    #k = np.array([[1,1,1],[1,-8,1],[1,1,1]])
    #dst = cv.filter2D(image,cv.CV_32F,kernel=k)
    lpls = cv.convertScaleAbs(dst)
    cv.imshow('lpls',lpls)

cv.imshow("orign",src)
#sobel_demo(src)
lpls_demo(src)
cv.waitKey(0)
cv.destroyAllWindows()