import numpy as np
import cv2 as cv

src = cv.imread("img\\water_coins.jpg")
cv.imshow("orign",src)

def distanceTransform_demo(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
    kernel = cv.getStructuringElement(cv.MORPH_RECT,(3,3))
    opening = cv.morphologyEx(binary, cv.MORPH_OPEN, kernel, iterations=2)
    res = cv.distanceTransform(opening,cv.DIST_L2,5)
    #归一化
    res_out = cv.normalize(res,0,1.0,cv.NORM_MINMAX)
    cv.imshow("res",res_out*50)

distanceTransform_demo(src)
def watershed_demo(image):
    gray = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
    ret,binary = cv.threshold(gray,0,255,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
    cv.imshow("binary",binary)
    #噪声去除
    kernel = cv.getStructuringElement(cv.MORPH_RECT,(3,3))
    opening = cv.morphologyEx(binary,cv.MORPH_OPEN,kernel,iterations=2)
    cv.imshow("opening", opening)
    #确定背景区域
    sure_bg = cv.dilate(opening,kernel,iterations=3)
    cv.imshow("bg",sure_bg)
    #寻找前景区域
    #距离变换
    dist_transform = cv.distanceTransform(opening,cv.DIST_L2,5)
    ret,sure_fg = cv.threshold(dist_transform,0.7*dist_transform.max(),255,cv.THRESH_BINARY)
    cv.imshow("fg",sure_fg)
    #找到未知区域
    sure_fg = np.uint8(sure_fg)
    unknown = cv.subtract(sure_bg,sure_fg)
    cv.imshow("unknown", unknown)

    # 类别标记
    ret, markers = cv.connectedComponents(sure_fg)
    # 为所有的标记加1，保证背景是0而不是1
    markers = markers + 1
    # 现在让所有的未知区域为0
    markers[unknown == 255] = 0
    #使用分水岭算法,然后标记图像将被修改。边界区域将标记为-1。
    markers = cv.watershed(image, markers)
    image[markers == -1] = [0, 0, 255]
    cv.imshow("res",image)


#watershed_demo(src)
cv.waitKey(0)
cv.destroyAllWindows()