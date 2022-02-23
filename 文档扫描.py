import numpy as np
import cv2 as cv
import math
from numpy.lib.type_check import imag
from scipy import ndimage
class Scanner:
    def __init__(self,img):
        self.img = img
    def scan_view(self):
        print("scanned View")
        image = cv.imread(self.img)
        image = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
        #ret,binary = cv.threshold(gray,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
        threshlod = cv.adaptiveThreshold(image,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY_INV,11,10)
        out = (image>threshlod).astype("uint8")*255
        cv.imshow("out_view",out)
        return out

    def Rotation(self):
        print("Rotation")
        image = cv.imread(self.img)
        image = cv.GaussianBlur(image,(3,3),0)
        image = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
        img_edges = cv.Canny(image,100,100)
        lines = cv.HoughLinesP(img_edges,rho=1,theta=np.pi/180.0,threshold=160,minLineLength=100,maxLineGap=10)
        angles = []
        for [[x1,y1,x2,y2]] in lines:
            #cv.line(image,(x1,y1),(x2,y2),(255,0,0),2)
            angle = math.degrees(math.atan2(y2-y1,x2-x1))
            angles.append(angle)
        median_angle = np.median(angles)
        image = ndimage.rotate(image,median_angle)
        cv.imshow("out_rotation",image)
        cv.imwrite("out_rotation.jpg",image)
        return image
if __name__=="__main__":
    img = "img\\test.jpg"
    sc = Scanner(img)
    sc.Rotation()
    img = "out_rotation.jpg"
    sc = Scanner(img)
    sc.scan_view()
    cv.waitKey(0)
    cv.destroyAllWindows

