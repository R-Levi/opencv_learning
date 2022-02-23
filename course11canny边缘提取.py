import cv2 as cv
import numpy as np

src = cv.imread('img\\lena.jpg')

def canny_demo(image):
    blured = cv.GaussianBlur(image,(3,3),0)
    gray = cv.cvtColor(blured,cv.COLOR_BGR2GRAY)
    gradx = cv.Sobel(gray,cv.CV_16SC1,1,0)
    grady = cv.Sobel(gray,cv.CV_16SC1,0,1)
    edge_output = cv.Canny(gradx,grady,50,150)
    cv.imshow("edge_output",edge_output)
    '''
    blured = cv.GaussianBlur(image,(3,3),0)
    out_blur = cv.Canny(blured,50,150)
    dst = cv.bitwise_and(image,image,mask=out_blur)
    cv.imshow('out_blur', dst)

    noise = image_noise(image)
    out_noise = cv.Canny(noise, 50, 150)
    cv.imshow("out_noise",out_noise)
    '''
def judge(x):
    if x>255:
        return 255
    if x<0:
        return 0
    return x
#生成高斯噪声
def image_noise(image):
    h,w,c = image.shape
    for i in range(h):
        for j in range(w):
            s = np.random.normal(0,20,3)#生成高斯分布的概率密度随机数
            b = judge(image[i,j,0]+s[0])
            g = judge(image[i,j,1]+s[1])
            r = judge(image[i,j,2]+s[2])
            image[i, j, 0] = b
            image[i, j, 1] = g
            image[i, j, 2] = r
    cv.imshow("image_noise",image)
    return image


cv.imshow("orign",src)
canny_demo(src)
cv.waitKey(0)
cv.destroyAllWindows()