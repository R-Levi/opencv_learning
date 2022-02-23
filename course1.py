#图像加载与保存
import cv2 as cv
import numpy as np
def get_image_info(image):
    print(type(image))
    print(image.shape)
    print(image.size)
    print(image.dtype)
    pixel_data = np.array(image)
    print(pixel_data)#像素数据

def video_demo():
    capture = cv.VideoCapture('test.mp4')
    while(True):
        ret,frame = capture.read()
        #frame = cv.flip(frame, 1) 镜像反转
        cv.imshow("video",frame)
        c = cv.waitKey(50)
        if c == 20:
            break

src = cv.imread('img\ghost.jpg')
get_image_info(src)
gray = cv.cvtColor(src,cv.COLOR_BGR2GRAY)#获取灰度图像
cv.imwrite('result.png',gray)
'''
<class 'numpy.ndarray'>
(312, 500, 3)
468000
uint8
'''