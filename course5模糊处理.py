import cv2 as cv
import numpy as np

#均值模糊
def blur_demo(image):
    img = cv.blur(image,(5,5))
    cv.imshow("blur",img)

#中值模糊 去椒盐噪声
def median_blur(image):
    #加入椒盐噪声
    for i in range(1000):
        x = np.random.randint(0,image.shape[0])
        y = np.random.randint(0,image.shape[1])
        image[x,y] = (0,255,255)
    cv.imshow('image_blur',image)
    img = cv.medianBlur(image,3)
    cv.imshow('median_blur',img)

def custom_blur(image):
    #kernel = np.ones([5,5],np.float32)/25
    #锐化核
    kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]],np.float32)
    img = cv.filter2D(image,-1,kernel)
    cv.imshow('custom_blur', img)

#高斯模糊
def judge(x):
    if x>255:
        return 255
    if x<0:
        return 0
    return x
#生成噪声
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
#高斯双边模糊
def bi_demo(image):
    img = cv.bilateralFilter(image,0,100,100)
    cv.imshow("bi_image",img)
#均值迁移模糊是图像边缘保留滤波算法中一种，经常用来在对图像进行分水岭分割之前去噪声，
#可以大幅度提升分水岭分割的效果。它的基本原理是：
#对于给定的一定数量样本，任选其中一个样本，以该样本为中心点划定一个圆形区域，
# 求取该圆形区域内样本的质心，即密度最大处的点，再以该点为中心继续执行上述迭代过程，
# 直至最终收敛。
#可以利用均值偏移算法的这个特性，实现彩色图像分割
def shift_demo(image):
    img = cv.pyrMeanShiftFiltering(image,10,50)
    cv.imshow("shift_image",img)


src = cv.imread('img\gxt.jpg')
cv.imshow("image",src)

'''
#blur_demo(src)
#median_blur(src)
#custom_blur(src)
#原图高斯模糊
blur1 = cv.GaussianBlur(src,(5,5),10)
cv.imshow("GaussianBlur1",blur1)
img = image_noise(src)
#高斯噪声图高斯模糊
blur2 = cv.GaussianBlur(img,(5,5),100)
cv.imshow("GaussianBlur2",blur2)
'''
bi_demo(src)
shift_demo(src)


cv.waitKey(0)
cv.destroyAllWindows()