import cv2 as cv
import numpy as np

img = cv.imread('img\\detect_blob.png')
cv.imshow('orign',img)

def hough_line(image):
    gray = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
    edges = cv.Canny(gray,50,150)
    #cv.imshow('edges',edges)
    lines = cv.HoughLines(edges,1.5,np.pi/180,200)
    for line in lines:
        rho,theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = rho*a
        y0 = rho*b
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv.line(image,(x1,y1),(x2,y2),(0,0,255),1)
    cv.imshow("hough_line_result",image)

def houghp_line(image):
    gray = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
    edges = cv.Canny(gray,50,150)
    #cv.imshow('edges',edges)
    lines = cv.HoughLinesP(edges,1,np.pi/180,100,minLineLength=100,maxLineGap=10)
    for line in lines:
        x1,y1,x2,y2 = line[0]
        cv.line(image,(x1,y1),(x2,y2),(0,0,255),1)
    cv.imshow("houghp_line_result",image)


def hough_circle(image):
    #滤波，对噪声敏感
    img = cv.medianBlur(image,5)
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    circles = cv.HoughCircles(gray,cv.HOUGH_GRADIENT,1,15,param1=80,param2=30,
                              minRadius=0,maxRadius=100)
    #print(circles)
    circles = np.uint16(np.around(circles))
    print(circles)
    for i in circles[0,:]:
        #绘制外圆
        cv.circle(img,(i[0],i[1]),i[2],(0,0,255),3)
        #绘制圆心
        cv.circle(img,(i[0],i[1]),2,(0,0,255),2)
    cv.imshow("result",img)



#houghp_line(image=img)
hough_circle(img)
cv.waitKey(0)
cv.destroyAllWindows()