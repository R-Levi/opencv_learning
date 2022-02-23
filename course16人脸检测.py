import cv2 as cv
import numpy as np



def face_detect_demo(image):
    gray = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
    #face_detector = cv.CascadeClassifier("D:\\opencv-4.5.2\\data\\haarcascades\\haarcascade_frontalface_alt_tree.xml")
    face_detector = cv.CascadeClassifier("D:\\opencv-4.5.2\\data\\lbpcascades\\lbpcascade_frontalface.xml")
    faces = face_detector.detectMultiScale(gray,1.10,4)
    for x,y,w,h in faces:
        cv.rectangle(image,(x,y),(x+w,y+h),(0,0,255),2)
    cv.imshow("result",image)

#src = cv.imread("img\\nba.jpg")
#cv.imshow("orign",src)
#face_detect_demo(src)
capture = cv.VideoCapture("img\\Megamind.avi")
while(True):
    ret,frame = capture.read()
    #frame = cv.flip(frame,1)
    face_detect_demo(frame)
    c = cv.waitKey(5)
    if c==27:
        break

cv.waitKey(0)
cv.destroyAllWindows()