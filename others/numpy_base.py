import cv2 as cv
import numpy as np

#像素反转
def acess_pixel(image):
    print(image.shape)
    height = image.shape[0]
    width = image.shape[1]
    channels = image.shape[2]#通道
    print('h:%s,w:%s,c:%s'%(height,width,channels))
    #循环每个像素
    for row in range(height):
        for col in range(width):
            for c in range(channels):
                pv = image[row,col,c]
                image[row,col,c] = 255-pv
    cv.imshow("pixel",image)
    #cv.waitKey(0)

def inverse(image):
    img = cv.bitwise_not(image)
    cv.imshow('img',img)
    cv.waitKey(0)

def creat_image():
    #三通道
    img = np.zeros([400,400,3],np.uint8)
    img[:,:,2] = np.ones([400,400])*255
    img[:,:,1] = np.ones([400,400])*244
    img[:,:,0] = np.ones([400,400])*25
    #0BLUE 1GREEN 2RED
    cv.imshow("new_img",img)
    cv.waitKey(0)

    # 单通道
    img0 = np.ones([400, 400, 1], np.uint8)
    img0 = img0*127
    cv.imshow("new_img0", img0)
    cv.waitKey(0)


#creat_image()

src = cv.imread('img\ghost.jpg')
t1 = cv.getTickCount()
inverse(src)
t2 = cv.getTickCount()
time = (t2-t1)/cv.getTickFrequency()
print("耗时:{}ms".format(time*1000))

tep = np.ones((3,3),np.float)
tep.fill(1222.656)#用标量值填充数组。
print(tep)

