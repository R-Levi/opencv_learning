import cv2 as cv
import numpy as np

src = cv.imread("img\\detect_blob.png")
cv.imshow("orign",src)


def measure_object(image):
    gray = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
    #二值化
    ret,binary = cv.threshold(gray, 0, 255, cv.THRESH_OTSU + cv.THRESH_BINARY_INV)
    print("threshold value:%s"%ret)
    cv.imshow("binary image",binary)
    dst = cv.cvtColor(binary,cv.COLOR_GRAY2BGR)
    #寻找轮廓
    outImg,contours,hireachy = cv.findContours(binary,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
    for i,contour in enumerate(contours):
        #计算轮廓面积，也可以用特征矩中的M00
        area = cv.contourArea(contour)
        print("rect area:%s"%area)
        #轮廓周长
        perimeter = cv.arcLength(contour, True)
        print("perimeter:%s"%perimeter)
        #轮廓的外包矩形,(x,y)为矩形的左上角坐标，（h,w）为举行宽高
        x,y,w,h = cv.boundingRect(contour)
        #宽高比
        rate = min(h,w)/max(h,w)
        print("rectangle rate:%s"%rate)
        #特征矩
        mm = cv.moments(contour)
        print(type(mm))#dict字典类型
        #轮廓重心
        cx = mm['m10']/mm['m00']
        cy = mm['m01']/mm['m00']
        #绘制中心黄色
        cv.circle(dst,(np.int(cx),np.int(cy)),3,(0,255,255),-1)
        #绘制矩形轮廓红色
        cv.rectangle(dst,(x,y),(x+w,y+h),(0,0,255),2)
        #轮廓近似
        approxCurve = cv.approxPolyDP(contour,4,True)
        print(approxCurve.shape)
        #绿色画出所有四条边的矩形
        if approxCurve.shape[0]==4:
            cv.drawContours(dst,contours,i,(0,255,0),2)
        #蓝色画出所有圆
        if approxCurve.shape[0]>=6:
            cv.drawContours(dst,contours,i,(255,0,0),2)
    cv.imshow("dst_image",dst)

def contours_demo(image):
    img = cv.GaussianBlur(image, (3, 3), 0)
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

    '''
    1.二值化
    ret,binary = cv.threshold(gray,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
    cv.imshow("binary",binary)
    '''

    '''
    2.canny边缘检测
    gradx = cv.Sobel(gray, cv.CV_16SC1, 1, 0)
    grady = cv.Sobel(gray, cv.CV_16SC1, 0, 1)
    edge_output = cv.Canny(gradx, grady, 50, 150)
    '''

    binary = cv.Canny(gray,30,100)

    clonImage,contours, hierarchy = cv.findContours(binary,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)

    cv.drawContours(image,contours,-1,(0,0,255),-1)
    '''
    for i,contour in enumerate(contours):
        cv.drawContours(image,contours,i,(0,0,255),2)
        print(i)
    '''
    cv.imshow("detect contours",image)


measure_object(src)
cv.waitKey(0)
cv.destroyAllWindows()