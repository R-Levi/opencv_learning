import numpy as np
import cv2 as cv

#src = cv.imread('img\\dog.jpg')
src = cv.imread('img\\big_image.jpg')
cv.imshow("origin",src)

#全局阈值
def threshold_demo(image):
    gray = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
    ret,binary = cv.threshold(gray,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
    #ret,binary = cv.threshold(gray,0,255,cv.THRESH_TOZERO_INV+cv.THRESH_TRIANGLE)
    print("threshold_value:%s"%ret)
    cv.imshow("binary",binary)

#自适应阈值（局部阈值）
def local_threshold_demo(image):
    gray = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
    binary = cv.adaptiveThreshold(gray,255,cv.ADAPTIVE_THRESH_MEAN_C,cv.THRESH_BINARY,25,10)
    binary = cv.adaptiveThreshold(gray,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,25,10)
    cv.imshow("local_threshold",binary)

def custom_threshold_demo(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    h,w = gray.shape[:2]
    m = np.reshape(gray,[1,h*w])
    mean = np.sum(m)/(h*w)
    ret,binary = cv.threshold(gray,mean,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
    cv.imshow("binary_custom",binary)

#超大图像的二值化，分块
def big_image(image):
    print(image.shape)#(439, 1200, 3)
    cw,ch = 128,128
    h,w = image.shape[:2]
    gray = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
    for row in range(0,h,ch):
        for col in range(0,w,cw):
            #分块
            roi = gray[row:row+cw,col:col+ch]
            '''
            #局部二值化
            binary = cv.adaptiveThreshold(roi,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,51,20)
            #全局二值化
            #ret,binary = cv.threshold(roi,0,255,cv.THRESH_OTSU+cv.THRESH_BINARY)
            '''
            #空白块过滤
            dev = np.std(roi)
            avg = np.mean(roi)
            if dev<10 or avg>220:
                gray[row:row + cw, col:col + ch] = 0
            else :
                ret, binary = cv.threshold(roi, 0, 255, cv.THRESH_OTSU + cv.THRESH_BINARY)
                gray[row:row + cw, col:col + ch] = binary
                print(np.std(binary),np.mean(binary))
    cv.imwrite('big_iamge_result.jpg',gray)

#threshold_demo(src)
#local_threshold_demo(src)
#custom_threshold_demo(src)

big_image(src)
cv.waitKey(0)
cv.destroyAllWindows()