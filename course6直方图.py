#histogram直方图
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
src = cv.imread('img\\lena.jpg')


def plot_demo(image):
    plt.hist(image.ravel(),256,[0,256])
    plt.show()

def image_hist(image):
    color = ('b','g','r')
    for i,c in enumerate(color):
        histr = cv.calcHist([image],[i],None,[256],[0,256])
        plt.plot(histr,color = c)
        plt.xlim([0,256])
    plt.show()

#直方图均衡化
#全局的均衡化
def equalHist_demo(image):
    gray = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
    equ = cv.equalizeHist(gray)
    res = np.hstack((gray,equ))
    cv.imshow("equ_hist",res)


#对比度受限的自适应直方图均衡化
def Clahe_Hist_demo(image):
    gray = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
    clahe = cv.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
    out = clahe.apply(gray)
    cv.imshow("clahe_hist",out)


#直方图的比较
def creat_hist_bgr(image):
    h,w,c = image.shape
    #设置bins = 16，降维
    rgbhist = np.zeros([16*16*16,1],np.float32)
    bsize = 16
    for row in range(h):
        for col in range(w):
            b = image[row,col,0]
            g = image[row,col,1]
            r = image[row,col,2]
            index = np.int(b/bsize)*16*16 + np.int(g/bsize)*16 + np.int(r/bsize)
            rgbhist[np.int(index), 0] += 1
    return rgbhist

def hist_compare(image1,image2):
    hist1 = creat_hist_bgr(image1)
    hist2 = creat_hist_bgr(image2)
    m1 = cv.compareHist(hist1,hist2,cv.HISTCMP_BHATTACHARYYA)#巴氏距离,值越小，相关度越高，最大值为1，最小值为0
    m2 = cv.compareHist(hist1,hist2,cv.HISTCMP_CORREL)#相关性,值越大，相关度越高，最大值为1，最小值为0
    m3 = cv.compareHist(hist1,hist2,cv.HISTCMP_CHISQR)#卡方,值越小，相关度越高，最大值无上界，最小值0
    print("巴氏距离: %s, 相关性: %s, 卡方: %s " % (m1, m2, m3))

#直方图反向投影
#HSV直方图
def hist2D_demo(image):
    hsv = cv.cvtColor(image,cv.COLOR_BGR2HSV)
    hist = cv.calcHist([hsv],[0,1],None,[30,32],[0,180,0,256])
    #cv.imshow("hist2D_demo",hist)
    plt.imshow(hist,interpolation='nearest')
    plt.title("hist2D_demo")
    plt.show()

def back_projection_demo():
    sample = cv.imread("img\\curry_target.jpg")
    s_hsv = cv.cvtColor(sample,cv.COLOR_BGR2HSV)
    target = cv.imread("img\\curry.jpg")
    t_hsv = cv.cvtColor(target, cv.COLOR_BGR2HSV)

    #cv.imshow("sample",sample)
    cv.imshow("target",target)

    roiHist = cv.calcHist([s_hsv],[0,1],None,[30,32],[0,180,0,256])
    cv.normalize(roiHist,roiHist,0,255,cv.NORM_MINMAX)
    dst = cv.calcBackProject([t_hsv],[0,1],roiHist,[0,180,0,256],1)
    dst_out = cv.merge((dst,dst,dst))
    res = cv.bitwise_and(target, dst_out)
    '''
     # 用圆盘进行卷积
    disc = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
    cv.filter2D(dst, -1, disc, dst)
    # 应用阈值作与操作
    ret, thresh = cv.threshold(dst, 70, 255, 0)
    thresh = cv.merge((thresh, thresh, thresh))
    res = cv.bitwise_and(target, thresh)
    '''
    cv.imshow("back_pro",res)

'''
image1 = cv.imread('img\\lena.jpg')
image2 = cv.imread('img\\lena_noise.jpg')
#归一化
cv.normalize(image1,image1,0,255,cv.NORM_MINMAX)
cv.normalize(image2,image2,0,255,cv.NORM_MINMAX)

cv.imshow("image1",image1)
cv.imshow("image2",image2)
hist_compare(image1,image2)
'''
#cv.imshow("image",src)
#equalHist_demo(src)
#Clahe_Hist_demo(src)
#hist2D_demo(src)
back_projection_demo()

cv.waitKey(0)
cv.destroyAllWindows()