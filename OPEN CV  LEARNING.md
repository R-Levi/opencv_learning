#  OPENCV  LEARNING  FOR BASE

reference---opencv中文网

[TOC]

![Screenshot_1](C:\Users\LEVI\Desktop\OPENCV\Screenshot_1.png)

## 安装遇到的问题：

1.conda镜像：.condarc文件中更换channels:（中科大镜像）

​    https://mirrors.ustc.edu.cn/anaconda/pkgs/main/

​	https://mirrors.ustc.edu.cn/anaconda/pkgs/free/

​	https://mirrors.ustc.edu.cn/anaconda/cloud/conda-forge/
​	ssl_verify: true

2.conda下载的opencv在pycharm无法使用：

​	①**环境不对：** 没有使用anaconda下的虚拟环境。

​	②**版本不匹配：** 路径选择对，OpenCV的版本不能被pycharm识别使用下面的语句进行更新

```conda
conda update --all
```

​	③使用pip install opencv-python安装，不使用conda 的opencv

3.使用opencv时pycharm 语法检查出现了cannot find reference '__init__' 

​	pip install opencv-contrib-python 安装这个

4.from .cv2 import *ImportError: numpy.core.multiarray failed to import

​	安装3.3版本的opencv-python时出现numpy版本问题，卸了重装即可

​	也有可能创建了numpy.py 与numpy命名冲突

5.conda pkgs缓存清理：

- 有些包安装之后，从来没有使用过；

- 一些安装包的tar包也保留在了计算机中；

- 由于依赖或者环境等原因，某些包的不同版本重复安装。

  **# conda clean使用**
   **## 删除从不使用的包**

  ```ruby
  $ conda clean --packages
  ```

  **## 删除tar包**

  ```ruby
  $ conda clean --tarballs
  ```

  **## 删除索引缓存、锁定文件、未使用过的包和tar包。**

  ```ruby
  $ conda clean -a 
  ```

```python
6.conda常用命令：
	管理环境: conda info -e
			conda info --envs
			conda env list
			conda会显示环境列表。
    创建环境：conda create -n 环境名 软件1=版本号 软件2=版本号。
    激活环境：activate 环境名。
    删除环境：conda remove -n 环境名 --all。
    搜索库：conda search 库名
    安装库：conda install 库名
```

```python
使用jupyter的扩展包:pip install jupyter_contrib_nbextensions
				jupyter contrib nbextension install --user
 启动 jupyter noteboook --ip 127.0.0.1 --port=8080
```



## 1.图像的加载保存

​		图像，结构化存储的数据信息，包括通道数目、高于宽、像素数据、图像类型

​    	加载：读取src = cv.imread('ghost.jpg')，有flag参数				

> ```html
> flags = -1：imread按解码得到的方式读入图像
> flags = 0：imread按单通道的方式读入图像，即灰白图像 
> flags = 1：imread按三通道方式读入图像，即彩色图像，默认参数
> ```

​					capture = cv.VideoCapture(0)  读取摄像头或者视频文件

​					ret,frame = capture.read()，读取每一帧图像

​		保存：cv.imwrite('result.png',src)

## 2.色彩空间

​	常见色彩空间，RGB、HSV(色调H（0-180）、饱和度S（0-255），明度V（0-255）)、HIS、YCrCb（皮肤检测）、YUV	

<img src="C:\Users\LEVI\Desktop\OPENCV\HSV.png" alt="HSV" style="zoom:80%;" />![image-20210615104546063](C:\Users\LEVI\AppData\Roaming\Typora\typora-user-images\image-20210615104546063.png)RGB

<img src="C:\Users\LEVI\Desktop\OPENCV\HSV_数值.png" alt="HSV_数值"  />

​		①色彩空间转换：

```python
cv.cvtColor(image,cv.COLOR_BGR2GRAY)
    灰度化常用公式：Gray=0.299R+0.587G+0.114B;
cv.cvtColor(image,cv.COLOR_BGR2HSV)
...
```

​		②inRange(InputArray src,InputArray lowerb,  InputArray upperb,OutputArray dst)：

​			参数1：输入要处理的图像，可以为单通道或多通道。

​			参数2：包含下边界的数组或标量。(对照数值图)

​			参数3：包含上边界数组或标量。(对照数值图)

​			参数4：输出图像，与输入图像src 尺寸相同且为CV_8U 类型

​			主要是将在两个阈值内的像素值设置为白色（255），而不在阈值区间内的像素值设置为黑色（0），该函数输出的dst是一幅二值化之后的图像。

​		③通道分离合并：

```python
b,g,r = cv.split(src)
src = cv.merge([b,g,r],src)
```

## 3.像素运算

算数运算

​		+、-、*、/，调节亮度，对比度等

```python
def add_demo(m1,m2):
    dst = cv.add(m1,m2)
    cv.imshow('add_demo',dst)

def sub_demo(m1,m2):
    dst = cv.subtract(m2,m1)
    cv.imshow('sub_demo',dst)


def divid_demo(m1,m2):
    dst = cv.divide(m1,m2)
    cv.imshow('divid',dst)

def multiply_demo(m1,m2):
    dst = cv.multiply(m1,m2)
    cv.imshow('mutiply',dst)

def others(m1,m2):
    #均值 方差
    h1,dev1 = cv.meanStdDev(m1)
    h2,dev2= cv.meanStdDev(m2)

#调整亮度对比度
def control_bright_demo(image,c,b):
    h,w,channels = image.shape
    blank = np.zeros([h,w,channels],image.dtype)
    #与黑的pic的进行不同比例的融合，达到调整对比度的目的
    dst = cv.addWeighted(image,c,blank,1-c,b)
    cv.imshow('dst',dst)
    
addWeighted函数：
参数1：src1，第一个原数组.
参数2：alpha，第一个数组元素权重
参数3：src2第二个原数组
参数4：beta，第二个数组元素权重
参数5：gamma，图1与图2作和后添加的数值。不要太大，不然图片一片白。总和等于255以上就是纯白色了。
参数6：dst，输出图片
```

逻辑运算

​		与或非，遮罩层控制

```python
#逻辑运算
def logic_demo(m1,m2):
    #dst = cv.bitwise_and(m1,m2)
    #dst = cv.bitwise_or(m1,m2)
    dst = cv.bitwise_not(m1,m2)
    cv.imshow('and',dst)
每个操作都有mask参数，用来支持掩膜操作，作用：
①提取感兴趣区,用预先制作的感兴趣区掩模与待处理图像相乘,得到感兴趣区图像,感兴趣区内图像值保持不变,而区外图像值都为0。
②屏蔽作用,用掩模对图像上某些区域作屏蔽,使其不参加处理或不参加处理参数的计算,或仅对屏蔽区作处理或统计。
③结构特征提取,用相似性变量或图像匹配方法检测和提取图像中与掩模相似的结构特征。
④特殊形状图像的制作。
example：
		src = cv.imread('img\dog.jpg')
		cv.imshow('orign_dog',src)
        hsv = cv.cvtColor(src,cv.COLOR_BGR2HSV)
        low = np.array([35,43,46])
        up = np.array([77,255,255])
        mask = cv.inRange(hsv,lowerb=low,upperb=up)
        cv.imshow('mask',mask)
        dst = cv.bitwise_and(src,src,mask=mask)
        cv.imshow('dst',dst)
        cv.waitKey(0)
        cv.destroyAllWindows()
```

效果：	

![mask_result](C:\Users\LEVI\Desktop\OPENCV\mask_result.png)

## 4.ROI与泛洪填充

​	ROI（region of interest），感兴趣区域。机器视觉、图像处理中，从被处理的图像以方框、圆、椭圆、不	规则多边形等方式勾勒出需要处理的区域，称为感兴趣区域，ROI。

```python
src = cv.imread('img\gxt.jpg')
cv.imshow('test',src)
face = src[25:180,130:255]#脸部的ROI
gray = cv.cvtColor(face,cv.COLOR_BGR2GRAY)
backface = cv.cvtColor(gray,cv.COLOR_GRAY2BGR)
src[25:180,130:255] = backface
cv.imshow('face',src)
结果：
```

​	<img src="C:\Users\LEVI\Desktop\OPENCV\ROI.png" alt="ROI" style="zoom: 50%;" />

泛洪填充:

```python
彩色图像填充
def fill_color_demo(image):
    copyImg = image.copy()
    h,w = image.shape[:2]
    mask = np.zeros([h+2,w+2],np.uint8)#+2是规定
    cv.floodFill(copyImg,mask=mask,seedPoint=(0,0),
                 newVal=(0,255,100),loDiff=(50,50,50),upDiff=(50,50,50),flags=cv.FLOODFILL_FIXED_RANGE)
    cv.imshow('fill_color',copyImg)
```

```python
二值图像填充
def fill_binary():
    image = np.zeros([400,400,3],np.uint8)
    image[100:300,100:300] = 255
    cv.imshow('img',image)

    mask = np.ones([402,402],np.uint8)
    mask[101:301,101:301]=0
    cv.floodFill(image,mask,(200,200),(255,255,0),cv.FLOODFILL_MASK_ONLY)
    cv.imshow('fill_binary',image)
    
```

```python
floodFill函数有以下参数：
	1.image参数表示输入/输出1或3通道，8位或浮点图像。
    2.mask参数表示掩码，该掩码是单通道8位图像，**比原图高度多2个像素，宽度多2个。填充时不能穿过输入掩码中的非零像素。**

​	   mask作用是限制填充的范围

​	3.seedPoint参数表示泛洪算法(漫水填充算法)的起始点。

​	4.newVal参数表示在重绘区域像素的新值。

​	5.loDiff参数表示当前观察像素值与其部件邻域像素值或待加入该组件的种子像素之间的亮度或颜色之负差的最大值。

　6.upDiff参数表示当前观察像素值与其部件邻域像素值或待加入该组件的种子像素之间的亮度或颜色之正差的最大值。

​	7.flags表示填充的类型：

​			设置FLOODFILL_FIXED_RANGE – 改变图像，泛洪填充   

​			设置FLOODFILL_MASK_ONLY – 不改变图像，只填充遮罩层本身，忽略新的颜色值参数

​	填充的范围是在mask为0的区域并且要填充的像素颜色范围在【seedpoint颜色-loDiff和seedpoint颜色+upDiff】之间
```



## 5.模糊处理

包括均值模糊（归一化滤波）、中值模糊、自定义模糊、高斯模糊

模糊操作的作用是在图片时减低噪声。

基本原理是基于离散卷积，定义好每一个卷积核，不同的卷积核得到不同的卷积效果，模糊是卷积的一种表象

### 1.均值模糊

```python
#均值模糊
def blur_demo(image):
    img = cv.blur(image,(5,5))
    cv.imshow("blur",img)
blur(src, ksize, dst=None, anchor=None, borderType=None)有两个重要的参数，一个是图像，另一个是卷积核大小，（5，5）表示5*5的卷积核，必须是奇数卷积核，通过计算这个范围内的均值来确定大小，传入（5，5）后均值模糊的卷积核就是:
		[1,1,1,1,1]
    	[1,1,1,1,1]
    1/25[1,1,1,1,1]
    	[1,1,1,1,1]
    	[1,1,1,1,1]
borderType=cv2.BORDER_CONSTANT，默认把虚线框里填充上0。
```

<img src="C:\Users\LEVI\Desktop\OPENCV\均值模糊.png" alt="均值模糊" style="zoom: 50%;" />

### 2.中值模糊

```python
#中值模糊 对于椒盐噪声有很好的效果
def median_blur(image):
    #加入椒盐噪声
    for i in range(1000):
        x = np.random.randint(0,image.shape[0])
        y = np.random.randint(0,image.shape[1])
        image[x,y] = (0,255,255)
    cv.imshow('image_blur',image)
    img = cv.medianBlur(image,3)
    cv.imshow('median_blur',img)
medianBlur与blur不同的是，ksize是一个奇数 表示ksize*ksize的卷积核，基本原理就是从原图中取出ksize*ksize个数，排序取中值，他的默认填充方式为BORDER_REPLICATE（aaaaaa|abcdefgh|hhhhhhh）
```

中值模糊结果：<img src="C:\Users\LEVI\Desktop\OPENCV\中值模糊.png" alt="中值模糊" style="zoom: 80%;" />

### 3.自定义模糊

```python
#自定义模糊
def custom_blur(image):
	#5*5的卷积核进行均值模糊
    #kernel = np.ones([5,5],np.float32)/25
    #锐化核实现图片的锐化处理
    kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]],np.float32)
    img = cv.filter2D(image,-1,kernel)
    cv.imshow('custom_blur', img)

filter2D(src, ddepth, kernel, dst=None, anchor=None, delta=None, borderType=None)
ddepth=-1表示目标图像[0,-1,0],[-1,5,-1],[0,-1,0]和原图像深度保持一致
[0,-1,0]
[-1,5,-1]是一个高通滤波器 实现锐化图像
[0,-1,0]
```

锐化结果：

![锐化](C:\Users\LEVI\Desktop\OPENCV\锐化.png)



### 4.**高斯模糊**

（基于权重的均值模糊），去噪方面比均值模糊更好

高斯模糊中将中心点作为原点，其他点按照其在高斯分布上的位置分配权重，获取到坐标矩阵后，根据σ计算出来权重矩阵，然后在除以权重之和，得到最终的权重矩阵。
**权重总值等于一，不会改变图片的亮度，大于1偏亮，小于1偏暗，等于0的话属于边缘检测核，可以把边缘转化为白色，非边缘转化为黑色**

```python
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
    #高斯模糊，对高斯噪声有抑制作用
  	out_GaussianBlur = cv.GaussianBlur(img,(5,5),15)
  	cv.imshow("out_GaussianBlur_image",out_GaussianBlur)

GaussianBlur(src, ksize, sigmaX, dst=None, sigmaY=None, borderType=None):
    src:输入图像
    ksize:卷积核大小是奇数
    sigmaX：X方向上的高斯核标准偏差
    sigmaY：Y方向上的高斯核标准偏差（sigmaY=0时，其值自动由sigmaX确定（sigmaY=sigmaX）；sigmaY=sigmaX=0时，它们的值将由ksize.width和ksize.height自动确定）
```

<img src="C:\Users\LEVI\Desktop\OPENCV\高斯模糊.png" alt="高斯模糊" style="zoom: 80%;" />

### 5.EPF边缘保留滤波

​		高斯模糊，只考虑了像素空间的分布，没有考虑像素值和另一个像素值之间差异的问题，像素间差异较大的情况下（比如图像的边缘），高斯模糊会进行处理，但是我们不需要处理边缘，要进行的操作就叫做边缘保留滤波（EPF），达到了保边去噪的效果。

颜色距离：

![双边颜色空间](C:\Users\LEVI\Desktop\OPENCV\双边颜色空间.png)

颜色距离和空间距离相乘得到最终的卷积模板：

![双边输出权重](C:\Users\LEVI\Desktop\OPENCV\双边输出权重.png)

颜色距离取决与被卷积像素的灰度值与领域的灰度值之间的差值，边缘有较大的灰度变化，就会使边缘和边缘的另一边区域生成较小的权值，与被卷积像素的灰度值类似的区域会生成比较大的权值。会有一个断崖的效果。

![双边过滤原理](C:\Users\LEVI\Desktop\OPENCV\双边过滤原理.png)

```python
#高斯双边模糊
def bi_demo(image):
    img = cv.bilateralFilter(image,0,90,25)
    cv.imshow("bi_image",img)    
bilateralFilter(src, d, sigmaColor, sigmaSpace, dst=None, borderType=None)：
	src：输入图像
	d：过滤过程中每个像素领域的直径。
    	过大的滤波器（d>5）执行效率低，对于实时应用，建议取 d=5；
		对于需要过滤严重噪声的离线应用，可取 d=9；
		d>0 时，由d指定邻域直径；
		d<=0 时，d会自动由sigmaSpace的值确定，且d与sigmaSpace 成正比
	sigmaColor：颜色空间滤波器的sigma值。参数越大，临近像素将会在越远的地方mix。
	sigmaSpace：坐标空间中滤波器的sigma值。参数值越大，意味着越远的像素会相互影响，使更大的区域足够相似的颜色获取相同的颜色。
    	对于两个σ，如果他们很小（小于10），那么滤波器几乎没有什么效果；他们很大（大于150），那么滤波器的效果会很强，使图像显得非常卡通化；
 
#图像均值迁移模糊
'''
均值迁移模糊是图像边缘保留滤波算法中一种，经常用来在对图像进行分水岭分割之前去噪声，可以大幅度提升分水岭分割的效果。它的基本原理是：
对于给定的一定数量样本，任选其中一个样本，以该样本为中心点划定一个圆形区域，求取该圆形区域内样本的质心，即密度最大处的点，再以该点为中心继续执行上述迭代过程，直至最终收敛。
可以利用均值偏移算法的这个特性，实现彩色图像分割
'''
def shift_demo(image):
    img = cv.pyrMeanShiftFiltering(image,10,50)#油画的效果
    cv.imshow("shift_image",img)
    
pyrMeanShiftFiltering(src, sp, sr, dst=None, maxLevel=None, termcrit=None)：
		sp：定义的漂移物理空间半径大小。
        sr：定义的漂移色彩空间半径大小。
        maxLevel：定义金字塔的最大层数。
        termcrit：定义的漂移迭代终止条件
'''
算法过程：
1. 迭代空间构建：以输入图像上src上任一点P0为圆心，建立物理空间上半径为sp，色彩空间上半径为sr的球形空间，物理空间上坐标2个值x、y，色彩空间上坐标3个值R、G、B（或HSV），构成一个5维的空间球体。其中物理空间的范围x和y是图像的长和宽，色彩空间的范围R、G、B分别是0~255。
2. 求取迭代空间的向量并移动迭代空间球体后重新计算向量，直至收敛：在1中构建的球形空间中，求得所有点相对于中心点的色彩向量之和后，移动迭代空间的中心点到该向量的终点，并再次计算该球形空间中所有点的向量之和，如此迭代，直到在最后一个空间球体中所求得的向量和的终点就是该空间球体的中心点Pn，迭代结束。
3. 更新输出图像dst上对应的初始原点P0的色彩值为本轮迭代的终点Pn的色彩值，如此完成一个点的色彩均值漂移。
4. 对输入图像src上其他点，依次执行步骤1,、2、3，遍历完所有点位后，整个均值偏移色彩滤波完成。
在这个过程中，关键参数是sp和sr的设置，二者设置的值越大，对图像色彩的平滑效果越明显，同时函数耗时也越多。
'''
```

![双边模糊](C:\Users\LEVI\Desktop\OPENCV\双边模糊.png)

## 6.直方图（histogram）

直方图可以总体了解图像的强度分布，X轴表示像素值（0-255），Y轴表示图像对应的像素值的数目，通过查看图像的直方图，您可以直观地了解该图像的对比度，亮度，强度分布等。

与直方图有关的术语：

> BINS：每个特征空间子区段的数目。，若每个像素一个，则BINS=256，若将整个直方图分成16个子部分，BINS=16
>
> DIMS：需要统计的特征的数目。例如：`dims=1`，表示我们仅统计灰度值。 
>
> RANGE：像素强度值的范围，通常是【0，256】，即所有强度

```python
#使用matplotlib绘制
def plot_demo(image):
    plt.hist(image.ravel(),256,[0,256])
    plt.show()

'''
几个关键的参数:
x:数据集，最终的直方图将对数据集进行统计
bins:统计区间的分布
range:显示区间
'''
```

对于BGR图使用cv.calcHist():

```python
def image_hist(image):
    color = ('b','g','r')
    for i,c in enumerate(color):#遍历数据对象
        histr = cv.calcHist([image],[i],None,[256],[0,256])
        plt.plot(histr,color = c)
        plt.xlim([0,256])
    plt.show()
    
cv.calcHist（images，channels，mask，histSize，ranges [，hist [，accumulate]]）:
1.images：它是uint8或float32类型的源图像。它应该放在方括号中，即“ [img]”。
2.channels：也以方括号给出。它是我们计算直方图的通道的索引。例如，如果输入为灰度图像，则其值为[0]。对于彩色图像，您可以传递[0]，[1]或[2]分别计算蓝色，绿色或红色通道的直方图。
3.mask：图像掩码。为了找到完整图像的直方图，将其指定为“无”。但是，如果要查找图像特定区域的直方图，则必须为此创建一个掩码图像并将其作为掩码。（我将在后面显示一个示例。）
4.histSize：这表示我们的BIN计数。需要放在方括号中。对于全尺寸，我们通过[256]。
5.ranges：这是我们的RANGE。通常为[0,256]。
```

<img src="C:\Users\LEVI\Desktop\OPENCV\直方图.png" alt="直方图" style="zoom: 50%;" />

图像中可以看出蓝色大多在高值域，显然是天空

**直方图的应用**

**1.直方图均衡化（调整对比度）**

通常情况下，暗图像直方图的分量集中在灰度较低的一段，而亮图像直方图分量集中在灰度较高的一端。如果一张图的灰度值分布近似均匀的话，这幅图较大的灰度动态范围和较高的对比度，图像的细节也更为丰富，**仅仅依靠输入图像的直方图信息，就可以得到一个变换函数，利用该变换函数可以将输入图像达到上述效果，该过程就是直方图均衡化**。

```python
#全局的均衡化
def equalHist_demo(image):
    gray = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
    equ = cv.equalizeHist(gray)
    cv.imshow("equ_hist",equ)
cv.equalizeHist(),它的输入只是灰度图像，输出是我们的直方图均衡图像

#对比度受限的自适应直方图均衡化
def Clahe_Hist_demo(image):
    gray = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
    clahe = cv.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
    out = clahe.apply(gray)
    cv.imshow("clahe_hist",out)
cv.createCLAHE():
    clipLimit:颜色对比度的阈值， 
    titleGridSize:进行像素均衡化的网格大小，即在多少网格下进行直方图的均衡化操作
    在这种情况下，图像被分成称为“tiles”的小块（在OpenCV中，tileSize默认为8x8）。然后，像往常一样对这些块中的每一个进行直方图均衡.因此，在较小的区域中，直方图将限制在一个较小的区域中（除非存在噪声）。如果有噪音，它将被放大。为了避免这种情况，应用了对比度限制。如果任何直方图bin超出指定的对比度限制（在OpenCV中默认为40），则在应用直方图均衡之前，将这些像素裁剪并均匀地分布到其他bin。均衡后，要消除图块边界中的伪影，请应用双线性插值。                                               ---OPENCV中文网
```

<img src="C:\Users\LEVI\Desktop\OPENCV\Hist均衡化.png" alt="Hist均衡化" style="zoom:50%;" />

全局的情况下，房屋部分由于亮度高，均衡化后房屋的一些信息消失了，局部均衡化的效果更佳。

**2.直方图的比较**

通过直方图可以比较两张图片的近似度，大概判断两张图是否相同，opencv中提供四种比较方法

<img src="C:\Users\LEVI\Desktop\OPENCV\hist比较.png" alt="hist比较" style="zoom:67%;" />

**[相关比较、卡方比较、十字交叉性(对于相似度比较这个算法不太好)、巴氏距离]**

<img src="C:\Users\LEVI\Desktop\OPENCV\Hist比较1.png" alt="Hist比较1" style="zoom:50%;" />

```python
#对上面的两个图进行直方图的比较
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
    
output：巴氏距离: 0.22839805367159988, 相关性: 0.9139885159456272, 卡方: 1383591.6287067865 
            
如果比较两个不同分辨率的图片，可以先进行归一化
cv.normalize(image1,image1,0,255,cv.NORM_MINMAX)
		src ：输入数组
		dst ：输出数组，数组的大小和原数组一致；
		alpha ：1,用来规范值，2.规范范围，并且是下限；
		beta ：只用来规范范围并且是上限；
		norm_type：归一化选择的数学公式类型；
		dtype：当小于零时，输出矩阵和输入矩阵有相同的数据类型；否则该值表示输出矩阵的数据类型
		mark ：掩码。选择感兴趣区域，选定后只能对该区域进行操作。
```

**3.直方图反向投影**

简单来说，它用于图像分割或在图像中查找感兴趣的对象

```python
#HSV直方图
def hist2D_demo(image):
    hsv = cv.cvtColor(image,cv.COLOR_BGR2HSV)
    hist = cv.calcHist([hsv],[0,1],None,[30,32],[0,180,0,256])
    #cv.imshow("hist2D_demo",hist)
    plt.imshow(hist,interpolation='nearest')
    plt.title("hist2D_demo")
    plt.show()
```

```python
#opencv 直方图反向投影
#OpenCV提供了一个内建的函数**cv.calcBackProject**()。它的参数几乎与**cv.calchist**()函数相同。它的一个参数是直方图，也就是物体的直方图，我们必须找到它。另外，在传递给backproject函数之前，应该对对象直方图进行归一化。
def back_projection_demo():
    sample = cv.imread("img\\curry_target.jpg")
    s_hsv = cv.cvtColor(sample,cv.COLOR_BGR2HSV)
    target = cv.imread("img\\curry.jpg")
    t_hsv = cv.cvtColor(target, cv.COLOR_BGR2HSV)

    cv.imshow("sample",sample)
    cv.imshow("target",target)
	
    roiHist = cv.calcHist([s_hsv],[0,1],None,[30,32],[0,180,0,256])
    cv.normalize(roiHist,roiHist,0,255,cv.NORM_MINMAX)

    dst = cv.calcBackProject([t_hsv],[0,1],roiHist,[0,180,0,256],1)
    cv.imshow("back_pro",dst)
bins少的时候效果明显，越多的话，细分的越多，效果不明显。
calcBackProject()参数说明：
	images:输入图像，图像深度必须位CV_8U,CV_16U或CV_32F中的一种，尺寸相同，每一幅图像都可以有任意的通道数
	channels:用于计算反向投影的通道列表，通道数必须与直方图维度相匹配，第一个数组的通道是从0到image[0].channels()-1,第二个数组通道从图像image[0].channels()到image[0].channels()+image[1].channels()-1计数
	hist:输入的直方图，直方图的bin可以是密集(dense)或稀疏(sparse)
	backProject:目标反向投影输出图像，是一个单通道图像，与原图像有相同的尺寸和深度
	ranges**:直方图中每个维度bin的取值范围
	scale=1:可选输出反向投影的比例因子
	uniform=true:直方图是否均匀分布(uniform)的标识符，有默认值true

```

<img src="C:\Users\LEVI\Desktop\OPENCV\back_projection.png" alt="back_projection" style="zoom:80%;" />



## 7.模板匹配

​		模式识别一类的，模板匹配是一种用于在较大图像中搜索和查找模板图像位置的方法。首先要有一个模板图像和一个待检测的图像。工作方式就是，在待检测的图像上，从左到右从上到下计算模板图像与重叠的子图像匹配度，程度越高，相同的可能性越大。

opencv中的模板匹配的算法：

​		CV_TM_SQDIFF 平方差匹配法，最好的匹配为0，值越大匹配越差
​		CV_TM_SQDIFF_NORMED 归一化平方差匹配法
​		CV_TM_CCORR 相关匹配法，采用乘法操作，数值越大表明匹配越好
​		CV_TM_CCORR_NORMED 归一化相关匹配法
​		CV_TM_CCOEFF 相关系数匹配法，最好的匹配为1，-1表示最差的匹配
​		CV_TM_CCOEFF_NORMED 归一化相关系数匹配法

![模板匹配1](C:\Users\LEVI\Desktop\OPENCV\模板匹配1.png)

<img src="C:\Users\LEVI\Desktop\OPENCV\模板匹配2.png" alt="模板匹配2" style="zoom:82%;" />

```python
def template_demo():
    tpl = cv.imread('img\\yao.png')
    target = cv.imread('img\\nba.jpg')
    cv.imshow("target",target)
    #使用三种匹配方式，标准方差、标准相关、标准相关系数匹配
    medthods = [cv.TM_SQDIFF_NORMED,cv.TM_CCORR_NORMED,cv.TM_CCOEFF_NORMED]
    #模板的宽高
    tw,th = tpl.shape[:2]
    for md in medthods:
        result = cv.matchTemplate(target,tpl,md)
        min_val,max_val,min_loc,max_loc = cv.minMaxLoc(result)
        print(cv.minMaxLoc(result))
        # 如果方法是TM_SQDIFF或TM_SQDIFF_NORMED，则取最小值
        if md==cv.TM_SQDIFF_NORMED:
            top_left = min_loc
        else:
            top_left = max_loc
        bottom_right = (tl[0]+tw,tl[1]+th)
        cv.rectangle(target,tl,br,(0,255,0),2)
        cv.imshow("match-"+np.str(md),target)
        #cv.imshow("match-"+np.str(md),result)
    
    
void cv::matchTemplate(
	cv::InputArray image, // 用于搜索的输入图像, 8U 或 32F, 大小 W-H
	cv::InputArray templ, // 用于匹配的模板，和image类型相同， 大小 w-h
	cv::OutputArray result, // 匹配结果图像, 类型 32F, 大小 (W-w+1)-(H-h+1)
	int method // 用于比较的方法
);
	函数cvMinMaxLoc(result,&min_val,&max_val,&min_loc,&max_loc,NULL);从result中提取最大值（相似度最高）以及最大值的位置（即在result中该最大值max_val的坐标位置max_loc，即模板滑行时左上角的坐标，类似于图中的坐标（x,y）。

```



![template_res](C:\Users\LEVI\Desktop\OPENCV\template_res.png)

**多对象的模板匹配**

假设您正在搜索具有多次出现的对象，则**cv.minMaxLoc**()不会为您提供所有位置，这种情况下，要使用阈值化。

```python
img_rgb = cv.imread('mario.png') 
img_gray = cv.cvtColor(img_rgb, cv.COLOR_BGR2GRAY) 
template = cv.imread('mario_coin.png',0) 
w, h = template.shape[::-1] 
res = cv.matchTemplate(img_gray,template,cv.TM_CCOEFF_NORMED) 
threshold = 0.8 
loc = np.where( res >= threshold) 
for pt in zip(*loc[::-1]):    
    cv.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2) 
    cv.imwrite('res.png',img_rgb)
```

## 8.图像二值化

​		彩色图像：三个通道0-255，0-255，0-255，所以可以有2^24位空间

​		灰度图像：一个通道0-255,所以有256种颜色

​		二值图像：只有两种颜色，黑和白，1白色，0黑色

### 1.全局二值化

​		1.**THRESH_BINARY**:过门限的值为最大值，其他值为0

​		2.**THRESH_BINARY_INV**:过门限的值为0，其他值为最大值

​		3.**THRESH_TRUNC**:过门限的值为门限值，其他值不变

​		4.**THRESH_TOZERO**:过门限的值不变，其他设置为0

​		5.**THRESH_TOZERO_INV**:过门限的值为0，其他不变

自动寻找阈值的方法：

​		1.OTSU（大律法）：cv.THRESH_BINARY+cv.THRESH_OTSU

​		OTSU算法的假设是存在阈值TH将图像所有像素分为两类C1(小于TH)和C2(大于TH)，则这两类像素各自的均值就为m1、m2，图像全局均值为mG。同时像素被分为C1和C2类的概率分别为p1、p2。

​		<img src="C:\Users\LEVI\Desktop\OPENCV\OSTU.png" alt="OSTU" style="zoom:80%;" />

​		2.三角形算法： cv.THRESH_TOZERO_INV+cv.THRESH_TRIANGLE

opencv在做二值化时，获取到灰度图像的直方图，找到波峰波谷连线，从直方图的各个点向连线作垂线，找到最长的，做一个offset偏移得出阈值 。  

```python
#全局阈值
def threshold_demo(image):
    gray = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
    ret,binary = cv.threshold(gray,0,255,cv.THRESH_TOZERO_INV+cv.THRESH_OTSU)
    #ret,binary = cv.threshold(gray,0,255,cv.THRESH_TOZERO_INV+cv.THRESH_TRIANGLE)
    print("threshold_value:%s"%ret)
    cv.imshow("binary",binary)

threshold(src, thresh, maxval, type, dst=None):
src参数表示输入图像（多通道，8位或32位浮点）。
thresh参数表示阈值。
maxval参数表示与THRESH_BINARY和THRESH_BINARY_INV阈值类型一起使用设置的最大值。
type参数表示阈值类型。
retval参数表示返回的阈值。若是全局固定阈值算法，则返回thresh参数值。若是全局自适应阈值算法，则返回自适应计算得出的合适阈值。
dst参数表示输出与src相同大小和类型以及相同通道数的图像。
返回值为ret阈值，binary二值化图像
```

还可以自己计算阈值：

```python
def custom_threshold_demo(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    h,w = gray.shape[:2]
    m = np.reshape(gray,[1,h*w])
    #算均值作为阈值
    mean = np.sum(m)/(h*w)
    ret,binary = cv.threshold(gray,mean,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
    cv.imshow("binary_custom",binary)
```

### 2.自适应阈值（局部阈值）

​		全局二值化在一些情况下并不好，比如光照亮度不一致时。在此，算法基于像素周围的小区域确定像素的阈值。因此，对于同一图像的不同区域，我们获得了不同的阈值，这为光照度变化的图像提供了更好的结果。

```python
#自适应阈值（局部阈值）
def local_threshold_demo(image):
    gray = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
    binary = cv.adaptiveThreshold(gray,255,cv.ADAPTIVE_THRESH_MEAN_C,cv.THRESH_BINARY,25,10)
    #binary = cv.adaptiveThreshold(gray,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,25,10)
    cv.imshow("local_threshold",binary)
    
adaptiveThreshold(src, maxValue, adaptiveMethod, thresholdType, blockSize, C, dst=None):
    src参数表示输入图像（多通道，8位或32位浮点）。
    maxval参数表示与THRESH_BINARY和THRESH_BINARY_INV阈值类型一起使用设置的最大值。
    adaptiveMethod:   cv.ADAPTIVE_THRESH_MEAN_C::阈值是邻近区域的平均值减去常数C。 						                               cv.ADAPTIVE_THRESH_GAUSSIAN_C:阈值是邻域值的高斯加权总和减去常数C。
    thresholdType：阈值类型，THRESH_BINARY 和THRESH_BINARY_INV两种   
    blockSize:确定附近区域的大小,必须奇数 
    C是从邻域像素的平均或加权总和中减去的一个常数。    
```

光强影响下的二值化，局部效果更佳：

![threshold](C:\Users\LEVI\Desktop\OPENCV\threshold.png)

对于超大图像的二值化，可以先进行图像的分块处理，然后使用局部二值化。

```python
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
    cv.imwrite('big_iamge_result3.jpg',gray)
```

局部二值化的效果更好。

## 9.图像金字塔

​		通常，我们过去使用的是恒定大小的图像。但是在某些情况下，我们需要使用不同分辨率的（相同）图像。例如，当在图像中搜索某些东西（例如人脸）时，我们不确定对象将以多大的尺寸显示在图像中。在这种情况下，我们将需要创建一组具有不同分辨率的相同图像，并在所有图像中搜索对象。这些具有不同分辨率的图像集称为“**图像金字塔**”

**1.高斯金字塔**

​		主要用来向下采样/降采样

步骤：

向下采样：

​		1.图像进行高斯模糊

​		2.将所有的偶数列去除

向上采样：

​		1.图像扩大两倍，新增的列和行以0填充

​		2.使用先前同样的内核（乘以4）与放大的图像卷积，获得新增像素的近似值		



```python
def pyramid_demo(image):
    level = 3
    temp = image.copy()
    pyramid_images = []
    for i in range(level):
        dst = cv.pyrDown(temp)
        pyramid_images.append(dst)
        cv.imshow("pyramid_down_"+str(i),dst)
        temp = dst.copy()
    return pyramid_images
```

<img src="C:\Users\LEVI\Desktop\OPENCV\pyrdown.png" alt="pyrdown" style="zoom: 80%;" />

**2.拉普拉斯金字塔**

​		用来从金字塔低层图像重建上层未采样图像，在数字图像处理中也即是预测残差，可以对图像进行最大程度的还原，配合高斯金字塔一起使用。

![lpls](C:\Users\LEVI\Desktop\OPENCV\lpls.png)

```python
def lpls_demo(image):
    pyramid_images = pyramid_demo(image)
    level = len(pyramid_images)
    for i in range(level-1,-1,-1):
        if i-1<0:
            expand = cv.pyrUp(pyramid_images[i],dstsize=image.shape[:2])
            lpls = cv.subtract(image,expand)
            cv.imshow('lpls_down_'+str(i),lpls)
        else:
            expand = cv.pyrUp(pyramid_images[i], dstsize=pyramid_images[i-1].shape[:2])
            lpls = cv.subtract(pyramid_images[i-1], expand)
            cv.imshow('lpls_down_' + str(i), lpls)
```

<img src="C:\Users\LEVI\Desktop\OPENCV\lplsup.png" alt="lplsup" style="zoom:80%;" />

## 10.图像梯度

​		在模糊处理一节中，我们用均值滤波器和中值滤波器等进行了图像的降噪，但是会带来图像模糊的副作用。清晰图片和模糊的图片之间的差别就是边缘轮廓的灰度变化是否强烈，强化轮廓特征就用到了图像梯度。

​		图像梯度是指图像某像素在x和y两个方向上的变化率（与相邻像素比较），是一个二维向量，由2个分量组成，X轴的变化、Y轴的变化 。

​		其中X轴的变化是指当前像素右侧（X加1）的像素值减去当前像素左侧（X减1）的像素值。

​		同理，Y轴的变化是当前像素下方（Y加1）的像素值减去当前像素上方（Y减1）的像素值。

​		计算出来这2个分量，形成一个二维向量，就得到了该像素的图像梯度。例如：

<img src="C:\Users\LEVI\Desktop\OPENCV\gradient.png" alt="gradient" style="zoom:67%;" />

### Sobel算子和Scharr算子

​		Sobel算子是普通一阶差分，是基于寻找梯度强度。Sobel算子用来计算图像灰度函数的近似梯度。Sobel算子根据像素点上下、左右邻点灰度加权差，在边缘处达到极值这一现象检测边缘。对噪声具有平滑作用，提供较为精确的边缘方向信息，边缘定位精度不够高。当对精度要求不是很高时，是一种较为常用的边缘检测方法。

​		该算子包含两组3x3的矩阵，分别为横向及纵向，将之与图像作平面卷积，即可分别得出横向及纵向的亮度差分近似值。如果以A代表原始图像，Gx及Gy分别代表经横向及纵向边缘检测的图像，其公式如下:

<img src="C:\Users\LEVI\Desktop\OPENCV\sobel算子.jpg" alt="sobel算子" style="zoom:67%;" />

然后依据梯度和角度计算公式计算图像的梯度。

​				                G = √(Gx^2+Gy^2) = |Gx|+|Gy|

Sobel具有平滑和微分的功效。即：Sobel算子先将图像横向或纵向平滑，然后再纵向或横向差分，得到的结果是平滑后的差分结果。

​		还可以通过参数ksize指定内核的大小。如果ksize = -1，则使用3x3 Scharr滤波器，比3x3 Sobel滤波器具有更好的结果，噪声比较敏感，需要降噪。

```python
def sobel_demo(image):
    # 获取x轴方向的梯度,对x求一阶导，一般图像都是256，CV_8U但是由于需要进行计算，为了避免溢出，所以我们选择CV_32F
    grad_x = cv.Sobel(image,cv.CV_32F,1,0)
    grad_y = cv.Sobel(image,cv.CV_32F,0,1)
    # 用convertScaleAbs()函数将其转回原来的uint8形式,转绝对值（转为单通道，0-255）
    gradx = cv.convertScaleAbs(grad_x)
    grady = cv.convertScaleAbs(grad_y)

    cv.imshow("grad_x",gradx)
    cv.imshow("grad_y",grady)
	
    #两个方向的梯度融合
    gradxy = cv.addWeighted(gradx,0.5,grady,0.5,0)
    cv.imshow("result",gradxy)
    
Sobel(src, ddepth, dx, dy, dst=None, ksize=None, scale=None, delta=None, borderType=None):
    ddepth：表示输出的图像的深度，ddepth =-1时，代表输出图像与输入图像相同的深度。
    dx:x方向的擦差分阶数。1或0
    dy:y方向的擦差分阶数。1或0 
    ksize:表示sobel算子大小，必须是1，3，5，7
    scale：表示缩放导数的比例
    delta：表示可选增量
    borderType：表示判断图像边界的模式。这个参数默认值为cv2.BORDER_DEFAULT。
convertScaleAbs(src, dst=None, alpha=None, beta=None):
    src：参数表示原数组。
	dst：参数表示输出数组 (深度为 8u)。
	alpha：参数表示比例因子。
	beta：参数表示原数组元素按比例缩放后添加的值。
addWeighted(src1, alpha, src2, beta, gamma, dst=None, dtype=None):
    src1参数表示需要加权（乘以权重再相加）的第一个输入数组。
	alpha参数表示第一个数组的权重。
	src2参数表示第二个输入数组，它和第一个数组拥有相同的尺寸和通道数。
 	beta参数表示第二个数组的权重。
	gamma参数表示一个加到权重总和上的标量值。
	dst参数表示输出的数组，它和输入的两个数组拥有相同的尺寸和通道数。
	dtype参数表示输出数组的可选深度。当两个输入数组具有相同的深度时，这个参数设置为-1（默认值），即等同于src1.depth（）。
   
```

下图为x，y以及融合后的图像：

<img src="C:\Users\LEVI\Desktop\OPENCV\sobel.png" alt="sobel" style="zoom: 50%;" />

### Laplacian算子（二阶差分）

​		它计算了由关系Δsrc=∂2src/∂x^2+∂2src/∂y2给出的图像的拉普拉斯图,它是每一阶导数通过Sobel算子计算。如果ksize = 1,然后使用以下内核用于过滤:

​                           		    			[0 1 0]   		  [1 1 1]

​							kernel 	= 	[1 -4 1]  或	 [1 -8  1]

​													[0 1 0]			[1 1  1]

```python
def lpls_demo(image):
    dst = cv.Laplacian(image,cv.CV_32F)
    lpls = cv.convertScaleAbs(dst)
    #k = np.array([[1,1,1],[1,-8,1],[1,1,1]])
    #dst = cv.filter2D(image,cv.CV_32F,kernel=k)
    lpls = cv.convertScaleAbs(dst)
    cv.imshow('lpls',lpls)
Laplacian(src, ddepth, dst=None, ksize=None, scale=None, delta=None, borderType=None):
    ddepth：表示输出的图像的深度，ddepth =-1时，代表输出图像与输入图像相同的深度。
    ksize: 表示用于计算二阶导数滤波器的孔径大小，大小必须是正数和奇数
    scale参数表示计算拉普拉斯算子值的比例因子，默认情况下没有伸缩系数。
	delta参数表示一个可选的增量，将会加到最终的dst中，同样，默认情况下没有额外的值加到dst中。
	borderType表示判断图像边界的模式。这个参数默认值为cv2.BORDER_DEFAULT。
```

<img src="C:\Users\LEVI\Desktop\OPENCV\Laplacian.png" alt="Laplacian" style="zoom:50%;" />

## 11.Canny边缘提取

​		Canny边缘检测是一种非常流行的边缘检测算法，是John Canny在1986年提出的。它是一个多阶段的算法，即由多个步骤构成。（canny算法对于噪声敏感）

**1.图像降噪**

​		我们知道梯度算子可以用于增强图像，本质上是通过增强边缘轮廓来实现的，也就是说是可以检测到边缘的。但是，它们受噪声的影响都很大。那么，我们第一步就是想到要先去除噪声，因为噪声就是灰度变化很大的地方，所以容易被识别为伪边缘。

**2.计算图像梯度**

​		计算图像梯度能够得到图像的边缘，因为梯度是灰度变化明显的地方，而边缘也是灰度变化明显的地方。当然这一步只能得到可能的边缘。因为灰度变化的地方可能是边缘，也可能不是边缘。这一步就有了所有可能是边缘的集合。

**3.非极大值抑制**

​		非极大值抑制。通常灰度变化的地方都比较集中，将局部范围内的梯度方向上，灰度变化最大的保留下来，其它的不保留，这样可以剔除掉一大部分的点。将有多个像素宽的边缘变成一个单像素宽的边缘。即“胖边缘”变成“瘦边缘”。

下图以3*3的领域内对应领域值的大小，蓝线表示梯度的方向，

![canny_非极大值抑制](C:\Users\LEVI\Desktop\OPENCV\canny_非极大值抑制.png)

**4.阈值筛选**

​		双阈值筛选。通过非极大值抑制后，仍然有很多的可能边缘点，进一步的设置一个双阈值，即低阈值（low），高阈值（high）。灰度变化大于high的，设置为强边缘像素，低于low的，剔除。在low和high之间的设置为弱边缘。进一步判断，如果其领域内有强边缘像素，保留，如果没有，剔除。这样做的目的是只保留强边缘轮廓的话，有些边缘可能不闭合，需要从满足low和high之间的点进行补充，使得边缘尽可能的闭合。



```python
def canny_demo(image):
    blured = cv.GaussianBlur(image,(3,3),0)
    out_blur = cv.Canny(blured,50,150)
    #利用逻辑与和mask变成彩色边缘图像，利用掩膜（mask）进行“与”操作，即掩膜图像白色区域是对需要处理图像像素的保留，黑色区域是		#对需要处理图像像素的剔除，其余按位操作原理类似只是效果不同而已。
	dst = cv.bitwise_and(image,image,mask=out_blur)
    cv.imshow('out_blur', dst)
    
    noise = image_noise(image)
    out_noise = cv.Canny(noise, 50, 150)
    cv.imshow("out_noise",out_noise)

def judge(x):
    if x>255:
        return 255
    if x<0:
        return 0
    return x
#生成高斯噪声
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
```

下面是有噪声和降噪时，canny算法处理结果,说明canny算法对噪声敏感。

<img src="C:\Users\LEVI\Desktop\OPENCV\canny.png" alt="canny" style="zoom:67%;" />

## 12.霍夫变换（Hough）

​		霍夫变换之前要先进行边缘检测。霍夫变换是一个特征提取技术。其可用于隔离图像中特定形状的特征的技术，应用在图像分析、计算机视觉和数字图像处理领域。

### 1.霍夫线变换

霍夫变换直线检测的原理：

​		直线的斜截式方程是 y=mx+c，如下图，直线的斜率为tan(Π-(Π/2-θ)) = -cot(θ),直线与y轴的焦点f(0) = c = r/sin(θ),带入笛卡尔坐标方程，y = -cot(θ)*x + r/sin(θ) = (-cos(θ)/sin(θ))x+r/sin(θ),变换为极坐标方程就是 r = xcosθ + ysinθ 。参数（r，θ）平面叫做霍夫空间。

​	                                                              	<img src="C:\Users\LEVI\Desktop\OPENCV\极坐标.png" alt="极坐标" style="zoom:67%;" />

​		我们可以得到一个结论，给定平面中的单个点，那么通过该点的所有直线的集合对应于(r,θ)平面中的正弦曲线，这对于该点是独特的。两个或更多点形成一条直线，对应于极坐标上，这几个点的正弦曲线会有一个交叉点。因此，检测共线点的问题可以转化为找到曲线的交点问题。



​		下面三个点，对每个点的不同的角度直线做变换后得到（r，θ）空间坐标。

​		                        <img src="C:\Users\LEVI\Desktop\OPENCV\Hough_ex1.png" alt="Hough_ex1" style="zoom: 80%;" />

​		3个点霍夫变换后形成的正弦曲线：

​												<img src="C:\Users\LEVI\Desktop\OPENCV\hough_ex2.png" alt="hough_ex2" style="zoom:67%;" />

​    	在60°的角度下，三条曲线相交，说明粉色直线是检测到的直线，越多曲线交于一点也就意味着这个交点表示的直线由更多的点组成. 一般来说我们可以通过设置直线上点的 **阈值** 来定义多少条曲线交于一点我们才认为 *检测* 到了一条直线。

​		这就是霍夫线变换要做的. 它追踪图像中每个点对应曲线间的交点. 如果交于一点的曲线的数量超过了 **阈值** ，那么可以认为这个交点所代表的参数对（r，θ）在原图像中为一条直线。

​		考虑一个`100x100`的图像，中间有一条水平线。取直线的第一点。您知道它的`(x，y)`值。现在在线性方程式中，将值θθ= 0,1,2，..... 180放进去，然后检查得到ρρ。对于每对(ρ，θ)(ρ，θ)，在累加器中对应的(ρ，θ)(ρ，θ)单元格将值增加1。所以现在在累加器中，单元格(50,90)= 1以及其他一些单元格。

​		现在，对行的第二个点。执行与上述相同的操作。递增(ρ，θ)对应的单元格中的值。这次，单元格`(50,90)=2`。实际上，您正在对(ρ，θ)值进行投票。您对线路上的每个点都继续执行此过程。在每个点上，单元格(50,90)都会增加或投票，而其他单元格可能会或可能不会投票。这样一来，最后，单元格(50,90)的投票数将最高。因此，如果您在累加器中搜索最大票数，则将获得(50,90)值，该值表示该图像中的一条线与原点的距离为50，角度为90度。在下面的动画中很好地显示了该图片。

​																![houghlinesdemo](C:\Users\LEVI\Desktop\OPENCV\houghlinesdemo.gif)

<img src="C:\Users\LEVI\Desktop\OPENCV\Hough算法步骤.png" alt="Hough算法步骤" style="zoom:80%;" />

```python
def hough_line(image):
    #转灰度图像
    gray = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
    #边缘检测
    edges = cv.Canny(gray,50,150)
    #cv.imshow('edges',edges)
    #霍夫线变换
    lines = cv.HoughLines(edges,1,np.pi/180,200)
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
    
HoughLines(image, rho, theta, threshold, lines=None, srn=None, stn=None, min_theta=None, max_theta=None): 
	rho : 参数极径 r 以像素值为单位的分辨率. 我们使用1像素.
	theta: 参数极角theta 以弧度为单位的分辨率. 我们使用1度 (即CV_PI/180)
	threshold: 要”检测” 一条直线所需最少的的曲线交点
	srn and stn: 参数默认为0. 查缺OpenCV参考文献来获取更多信息.
    lines:输出直线vector(ρ,θ) or (ρ,θ,votes) ρ是距坐标原点的距离， θ是以弧度表示的线旋转角度(0∼垂直直线,π/2∼水平直线) 
votes 曲线交点累加计数

```

<img src="C:\Users\LEVI\Desktop\OPENCV\hough_lines_res.png" alt="hough_lines_res" style="zoom: 67%;" />

```python
#概率霍夫变换
在霍夫变换中，您可以看到，即使对于带有两个参数的行，也需要大量计算。概率霍夫变换是我们看到的霍夫变换的优化。它没有考虑所有要点。取而代之的是，它仅采用随机的点子集，足以进行线检测。只是我们必须降低阈值。
def houghp_line(image):
    gray = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
    edges = cv.Canny(gray,50,150)
    #cv.imshow('edges',edges)
    lines = cv.HoughLinesP(edges,1,np.pi/180,100,minLineLength=100,maxLineGap=10)
    for line in lines:
        x1,y1,x2,y2 = line[0]
        cv.line(image,(x1,y1),(x2,y2),(0,0,255),1)
    cv.imshow("houghp_line_result",image)

- minLineLength：最小行长。小于此长度的线段将被拒绝。
- maxLineGap ：线段之间允许将它们视为一条线的最大间隙。
它直接返回行的两个端点。
```

<img src="C:\Users\LEVI\Desktop\OPENCV\houghlinesp_res.png" alt="houghlinesp_res" style="zoom:61%;" />

### 2.霍夫圆变换

​		霍夫圆变换同样是在hough空间进行累加”投票“，选择局部最大值生成候选圆。

​		二维图像中，圆的表示为：(x-a)^2 + (y-b)^2 = r^2; 其中(a,b)为圆心，r为半径。转化为极坐标参数方程：

​				a = x - rcos(θ)

​				b = y - rsin(θ)

经过每一个像素点的所有圆的方程都可以根据极坐标方程求出，每一个圆对应一个（a，b，r）参数对，画出曲线图为三维的曲线，多个点的三维曲线图交于一点说明该点为一个候选圆。这和线变换是类似的。但是三维的计算量和内存消耗是巨大的，opencv采用了Hough梯度法进行了算法的优化。

**Hough梯度法的原理过程：**

​	**Ⅰ估计圆心：**

​		1.原图进行canny边缘检测，得到二值图像

​		2.原图执行一次Sobel算子，计算所有像素的邻域梯度值

​		3.初始化圆心空间 N（a，b），令 所有N（a，b）= 0

​		4.遍历二值图中的所有非零像素点，沿着梯度方向（切线的垂直方向）画线，将线段经过的所有累加器的点（a，b）的N（a，b）+=1

![Hough梯度1](C:\Users\LEVI\Desktop\OPENCV\Hough梯度1.png)

​		5.统计排序N（a，b），得到可能的圆心

​	**Ⅱ估计半径（针对某一个圆心）**

​		1.计算canny图中的所有非零的距离圆心的距离

​		2.距离从小到大排序，根据阈值，选取合适的可能半径（比如3和3.5都选为3）

​		3.初始化半径空间r，N（r）=0

​		4.遍历canny图中的非零点，对于点所满足的半径r，N（r）+=1

​		5.统计可能的半径N（r）值越大，说明这个半径越可能是真正的半径

**霍夫梯度法缺点**

1. 在霍夫梯度法中，我们使用 Sobel 导数来计算局部梯度，那么随之而来的假设是，其可以视作等同于一条局部切线，并这个不是一个数值稳定的做法。在大多数情况下，这样做会得到正确的结果，但或许会在输出中产生一些噪声。
2. 在边缘图像中的整个非0像素集被看做每个中心的候选部分。因此，如果把累加器的阈值设置偏低，算法将要消耗比较长的时间。
3. 因为每一个中心只选择一个圆，如果有同心圆，就只能选择其中的一个。
4. 因为中心是按照其关联的累加器值的升序排列的，并且如果新的中心过于接近之前已经接受的中心的话，就不会被保留下来。且当有许多同心圆或者是近似的同心圆时，霍夫梯度法的倾向是保留最大的一个圆。可以说这是一种比较极端的做法，因为在这里默认Sobel导数会产生噪声，若是对于无穷分辨率的平滑图像而言的话，这才是必须的。

```python
def hough_circle(image):
    #滤波，对噪声敏感
    img = cv.medianBlur(image,5)
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    circles = cv.HoughCircles(gray,cv.HOUGH_GRADIENT,1,20,param1=50,param2=30,
                              minRadius=0,maxRadius=0)
    circles = np.uint16(np.around(circles))
    for i in circles[0,:]:
        #绘制外圆
        cv.circle(img,(i[0],i[1]),i[2],(255,255,255),2)
        #绘制圆心
        cv.circle(img,(i[0],i[1]),2,(0,0,255),2)
    cv.imshow("result",img)
    
HoughCircles(image, method, dp, minDist, circles=None, param1=None, param2=None, minRadius=None, maxRadius=None): 
    image：8bit单通道灰度图像
    method:hough变换方式，HOUGH_GRADIENT霍夫梯度法。
    dp:累加器图像分辨率，这个参数允许创建一个比输入图像分辨率低的累加器。（这样做是因为有理由认为图像中存在的圆会自然降低到与图像宽高相同数量的范畴）。如果dp设置为1，则分辨率是相同的；如果设置为更大的值（比如2），累加器的分辨率受此影响会变小（此情况下为一半）。dp的值不能比1小。：该参数是让算法能明显区分的两个不同圆之间的最小距离。
    minDist：该参数是让算法能明显区分的两个不同圆之间的最小距离。
    param1：用于Canny的边缘阀值上限，下限被置为上限的一半。
    param2：累加器的阀值。
    minRadius：最小圆半径
    maxRadius：最大圆半径    
    
```

<img src="C:\Users\LEVI\Desktop\OPENCV\Hough_circles .png" alt="Hough_circles " style="zoom:67%;" />

## 13.轮廓

### 1.检测轮廓

轮廓可以简单地解释为连接具有相同颜色或强度的所有连续点（沿边界）的曲线。轮廓是用于形状分析以及对象检测和识别的有用工具。

- 为了获得更高的准确性，请使用二进制图像。因此，在找到轮廓之前，请应用**阈值或canny边缘检测。**
- 从OpenCV 3.2开始，**findContours**()不再修改源图像。
- 在OpenCV中，找到轮廓就像从黑色背景中找到白色物体。因此请记住，要找到的对象应该是白色，背景应该是黑色。

```python
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
    #opencv2返回两个值：contours：hierarchy。opencv3会返回三个值,分别是img, countours, hierarchy
    clonImage,contours, hierarchy = cv.findContours(binary,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
    cv.drawContours(image,contours,-1,(0,0,255),2)
    #contours是一个python的list
    '''
    for i,contour in enumerate(contours):
        cv.drawContours(image,contours,i,(0,0,255),2)
        print(i)
    '''
    cv.imshow("detect contours",image)
    
    
findContours(image, mode, method, contours=None, hierarchy=None, offset=None):
	函数中有三个参数，第一个是源图像，第二个是轮廓检索模式，第三个是轮廓逼近方法。
		轮廓检索模式
        	CV_RETR_EXTERNAL：只检测最外围轮廓，包含在外围轮廓内的内围轮廓被忽略。
			CV_RETR_LIST：检测所有的轮廓，包括内围、外围轮廓，但是检测到的轮廓不建立等级关系，彼此之间独立，没有等	级关系，这就意味着这个检索模式下不存在父轮廓或内嵌轮廓**，所以hierarchy向量内所有元素的第3、第4个分量都会被置为-1。
    		CV_RETR_CCOMP: 检测所有的轮廓，但所有轮廓只建立两个等级关系，外围为顶层，若外围内的内围轮廓还包含了其他的轮廓信息，则内围内的所有轮廓均归属于顶层。
			CV_RETR_TREE: 检测所有轮廓，所有轮廓建立一个等级树结构。外层轮廓包含内层轮廓，内层轮廓还可以继续包含内嵌轮廓。     		轮廓逼近方法：
			cv::CHAIN_APPROX_NONE：将轮廓中的所有点的编码转换成点。
			cv::CHAIN_APPROX_SIMPLE：压缩水平、垂直和对角直线段，仅保留它们的端点。
			cv::CHAIN_APPROX_TC89_L1 or cv::CHAIN_APPROX_TC89_KCOS：应用Teh-Chin链近似算法中的一种风格。

drawContours(image, contours, contourIdx, color, thickness=None, lineType=None, hierarchy=None, maxLevel=None, offset=None):
    只要有边界点，它也可以用来绘制任何形状。
    它的第一个参数是源图像，第二个参数是应该作为Python列表传递的轮廓，第三个参数是轮廓的索引（在绘制单个轮廓时有用。要绘制所有轮廓，请传递-1），thickness取-1，表示填充轮廓，其余参数是颜色，厚度等等。
```



<img src="C:\Users\LEVI\Desktop\OPENCV\contours_res.png" alt="contours_res" style="zoom: 50%;" />

蓝色椭圆可能是在二值化时蓝色区域低于门限值，给过滤掉了。使用canny边缘检测时把阈值设置的低一点就可以。



​		findContours（）函数输出的hierarchy是一个向量，向量内每个元素都是一个包含4个int型的数组。向量hierarchy内的元素和轮廓向量contours内的元素是一一对应的，向量的容量相同。hierarchy内每个元素的4个int型变量是hierarchy[i][0] ~ hierarchy[i][3]，分别表示当前轮廓 i 的后一个轮廓、前一个轮廓、父轮廓和内嵌轮廓的编号索引。如果当前轮廓没有对应的后一个轮廓、前一个轮廓、父轮廓和内嵌轮廓，则相应的hierarchy[i][*]被置为-1。详细解释参考http://www.woshicver.com/FifthSection/4_9_5_%E8%BD%AE%E5%BB%93%E5%88%86%E5%B1%82/或者官网。

### 2.轮廓特征

轮廓的不同特征包括例如面积，周长，质心，边界框等。

```
OpenCV提取轮廓之后，还可以进行许多操作：

​	ArcLength() 计算轮廓长度 
​	ContourArea() 计算轮廓区域的面积 
​	BoundingRect() 轮廓的外包矩形 
​	ConvexHull() 提取轮廓的凸包 
​	IsContourConvex() 测试轮廓的凸性 
​	MinAreaRect() 轮廓的最小外包矩形 
​	MinEnclosingCircle() 轮廓的最小外包圆
​	fitEllipse()用椭圆拟合二维点集
​	approxPolyDP()逼近多边形曲线,根据我们指定的精度，它可以将轮廓形状近似为顶点数量较少的其他形状。它是Douglas-Peucker算法的实现
```

**特征距：**

​		矩函数在图像分析中有着广泛的应用，如模式识别、目标分类、目标识别与方位估计、图像的编码与重构等。从一幅图像计算出来的矩集，不仅可以描述图像形状的全局特征，而且可以提供大量关于该图像不同的几何特征信息，如大小，位置、方向和形状等。图像矩这种描述能力广泛应用于各种图像处理、计算机视觉和机器人技术领域的目标识别与方位估计中。

**一阶矩：与形状有关；**

**二阶矩：显示曲线围绕直线平均值的扩展程度；**

**三阶矩：关于平均值的对称性测量；由二阶矩和三阶矩可以导出7个不变矩。而不变矩是图像的统计特性，满足平移、伸缩、旋转均不变的不变性、在图像识别领域得到广泛的应用。**

![矩特征](C:\Users\LEVI\Desktop\OPENCV\矩特征.png)

![矩特征2](C:\Users\LEVI\Desktop\OPENCV\矩特征2.png)

```python
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
        #蓝色画出圆或者大于6条边的多边形
        if approxCurve.shape[0]>=6:
            cv.drawContours(dst,contours,i,(255,0,0),2)
    cv.imshow("dst_image",dst)
    
关于轮廓近似approxPolyDP(curve, epsilon, closed, approxCurve=None):
    epsilon：它是从轮廓到近似轮廓的最大距离。它是一个精度参数。
    closed：若为true，则说明近似曲线是闭合的；反之，若为false，则断开。
    approxCurve：输出的点集，当前点集是能最小包容指定点集的。画出来即是一个多边形。
```

<img src="C:\Users\LEVI\Desktop\OPENCV\轮廓_res.png" alt="轮廓_res" style="zoom:50%;" />

### 3.轮廓属性

**1. 长宽比**

它是对象边界矩形的宽度与高度的比值。

AspectRatio=WidthHeightAspectRatio=WidthHeight

```python
x,y,w,h = cv.boundingRect(cnt)
aspect_ratio = float(w)/h
```

**2.范围**

范围是轮廓区域与边界矩形区域的比值。

Extent=ObjectAreaBoundingRectangleAreaExtent=ObjectAreaBoundingRectangleArea

```python
area = cv.contourArea(cnt)
x,y,w,h = cv.boundingRect(cnt)
rect_area = w*h
extent = float(area)/rect_area
```

**3.坚实度**

坚实度是等高线面积与其凸包面积之比。

```python
Solidity=ContourAreaConvexHullAreaSolidity=ContourAreaConvexHullArea
area = cv.contourArea(cnt)
hull = cv.convexHull(cnt)
hull_area = cv.contourArea(hull)
solidity = float(area)/hull_area
```

**4.等效直径**

等效直径是面积与轮廓面积相同的圆的直径。

EquivalentDiameter=√4×ContourAreaπEquivalentDiameter=4×ContourAreaπ

```python
area = cv.contourArea(cnt)
equi_diameter = np.sqrt(4*area/np.pi)
```

**5.取向**

取向是物体指向的角度。以下方法还给出了主轴和副轴的长度。

```
(x,y),(MA,ma),angle = cv.fitEllipse(cnt)
```

**6.掩码和像素点**

在某些情况下，我们可能需要构成该对象的所有点。可以按照以下步骤完成：

```python
mask = np.zeros(imgray.shape,np.uint8)
cv.drawContours(mask,[cnt],0,255,-1)
pixelpoints = np.transpose(np.nonzero(mask))
#pixelpoints = cv.findNonZero(mask)
```

这里提供了两个方法，一个使用Numpy函数，另一个使用OpenCV函数(最后的注释行)。结果也是一样的，只是略有不同。Numpy给出的坐标是`(行、列)`格式，而OpenCV给出的坐标是`(x,y)`格式。所以基本上答案是可以互换的。注意，`row = x, column = y`。

**7.最大值，最小值和它们的位置**

我们可以使用掩码图像找到这些参数。

```python
min_val, max_val, min_loc, max_loc = cv.minMaxLoc(imgray,mask = mask)
```

**8.平均颜色或平均强度**

在这里，我们可以找到对象的平均颜色。或者可以是灰度模式下物体的平均强度。我们再次使用相同的掩码进行此操作。

```python
mean_val = cv.mean(im,mask = mask)
```

**9.极端点**

极点是指对象的最顶部，最底部，最右侧和最左侧的点。

```python
leftmost = tuple(cnt[cnt[:,:,0].argmin()][0])
rightmost = tuple(cnt[cnt[:,:,0].argmax()][0])
topmost = tuple(cnt[cnt[:,:,1].argmin()][0])
bottommost = tuple(cnt[cnt[:,:,1].argmax()][0])
```

## 14.形态学变换

​		**Morphological Operations：http://homepages.inf.ed.ac.uk/rbf/HIPR2/morops.htm at HIPR2**

​		图像形态学是图像处理学科的一个单独的分支学科，灰度和二值图处理中重要的手段，由数学集合论发展起来的一个学科

### 1.膨胀与腐蚀

**1.膨胀**

![膨胀](C:\Users\LEVI\Desktop\OPENCV\膨胀.png)

​		它恰好与侵蚀相反。 这里，如果内核下的至少一个像素为“1”，则像素元素为“1”. 因此它增加了图像中的白色区域或前景对象的大小增加，使用3×3的模板进行或操作，如果为1中心填充1，否则填0。

​		**它可用于连接对象的破碎部分**

**2.腐蚀**

![腐蚀](C:\Users\LEVI\Desktop\OPENCV\腐蚀.png)

​		腐蚀的基本思想：侵蚀前景物体的边界（总是试图保持前景为白色）；内核在图像中滑动（如在2D卷积中）.只有当内核下的所有像素都是1时，原始图像中的像素（1或0）才会被认为是1，否则它会被侵蚀（变为零）。使用3×3的模板进行与操作，如果为1中心填充1，否则填0。

​		它**有助于消除小的白噪声，分离两个连接的对象。**



```python
#腐蚀
def erode_demo(image):
    print(image.shape)
    gray = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
    ret,binary = cv.threshold(gray,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
    #内核,getStructuringElement函数会返回指定形状和尺寸的结构元素。
    #矩形：MORPH_RECT;交叉形：MORPH_CROSS;椭圆形：MORPH_ELLIPSE;
    kernel = cv.getStructuringElement(cv.MORPH_RECT,(3,3))
    print(kernel)
    dst = cv.erode(binary,kernel)
    cv.imshow("erode_image",dst)

#膨胀
def dilate_demo(image):
    print(image.shape)
    gray = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
    ret,binary = cv.threshold(gray,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
    #内核,getStructuringElement函数会返回指定形状和尺寸的结构元素。
    #矩形：MORPH_RECT;交叉形：MORPH_CROSS;椭圆形：MORPH_ELLIPSE;
    kernel = cv.getStructuringElement(cv.MORPH_RECT,(3,3))
    print(kernel)
    dst = cv.dilate(binary,kernel)
    cv.imshow("dilate_image",dst)
```

![形态学](C:\Users\LEVI\Desktop\OPENCV\形态学.png)

### 2.开闭操作

基于膨胀与腐蚀操作的组合形成。

开操作：**侵蚀然后扩张**的另一个名称。它对于消除噪音很有用。

闭操作：闭运算与开运算相反，**先扩张然后再侵蚀**。在关闭前景对象内部的小孔或对象上的小黑点时很有用。

开闭才做还可以进行水平或者垂直线的提取，这个跟选取的开闭操作的结构元素有关。

```python
#开操作
def open_demo(image):
    print(image.shape)
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    dst = cv.morphologyEx(binary,cv.MORPH_OPEN,kernel)
    cv.imshow("open_image", dst)

#闭操作
def close_demo(image):
    print(image.shape)
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    dst = cv.morphologyEx(binary,cv.MORPH_CLOSE,kernel)
    cv.imshow("close_image", dst)

morphologyEx(src, op, kernel, dst=None, anchor=None, iterations=None, borderType=None, borderValue=None):
    op：为形态变换的类型，包括如下取值类型：
    	MORPH_ERODE：腐蚀，当调用morphologyEx使用MORPH_ERODE类型时，与调用腐蚀函数erode效果相同。
		MORPH_DILATE：膨胀，当调用morphologyEx使用MORPH_DILATE类型时，与调用膨胀函数dilate效果相。
		MORPH_OPEN：开运算，对图像先进行腐蚀再膨胀，等同于dilate(erode(src,kernal))，开运算对图像的边界进行平滑、去掉凸起等。
		MORPH_CLOSE：闭运算，对图像先进行膨胀在腐蚀，等同于erode(dilate(src,kernal))，闭运算用于填充图像内部的小空洞、填充图像的凹陷等。
		MORPH_GRADIENT：梯度图，用膨胀图减腐蚀图，等同于dilate(src,kernal)−erode(src,kernal)，可以用于获得图像中物体的轮廓。
		MORPH_TOPHAT：顶帽，又称礼帽，用原图像减去开运算后的图像，等同于src−open(src,kernal)，可以用于获得原图像中比周围亮的区域。
		MORPH_BLACKHAT：黑帽，闭运算图像减去原图像，等同于close(src,kernal)−src，可以用于获取原图像中比周围暗的区域。
		MORPH_HITMISS：击中击不中变换，用于匹配处理图像中是否存在核对应的图像，匹配时，需要确保核矩阵非0部分和为0部分都能匹配，注意该变换只能处理灰度图像。
	kernel:输入一个数组作为核。
	anchor:核的锚点位置，负值说明该锚点位于核中心。默认为核中心。
	iterations:整型int。腐蚀与膨胀被应用的次数。默认为None。
```

![开闭操作](C:\Users\LEVI\Desktop\OPENCV\开闭操作.png)

左侧为闭操作，先膨胀后腐蚀，去除了j中黑色噪点。

右侧为开操作，先腐蚀后膨胀，去吃了图片中的白色小点。

```python
去除水平线
# 核设定为以下
kernel = cv.getStructuringElement(cv.MORPH_RECT, (1, 15))
去除竖线
# 核设定为以下
kernel = cv.getStructuringElement(cv.MORPH_RECT, (15, 1))
```

![delete_vertical and horiztal line](C:\Users\LEVI\Desktop\OPENCV\delete_vertical and horiztal line.png)

### 3.其他形态学

**顶帽：**原图像和开操作之间的差值图像

**黑帽：**原图像和闭操作之间的差值图像

**形态学梯度：**

​		基本梯度：膨胀后的图像减去腐蚀后的图像得到的差值。opencv中支持的梯度。

​		内部梯度：原图像减去腐蚀之后的图像得到的差值图像。

​		外部梯度：原图像膨胀后减去原图像得到的差值图像。

```python
#顶帽
def top_hat_demo(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    dst = cv.morphologyEx(gray,cv.MORPH_TOPHAT,kernel)
    cv.imshow("top_hat_image", dst)
结果为下图右侧。
    
#黑帽
def black_hat_demo(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    dst = cv.morphologyEx(gray,cv.MORPH_BLACKHAT,kernel)
    cv.imshow("black_hat_image", dst)
结果为下图左侧。
```

![顶帽和黑帽](C:\Users\LEVI\Desktop\OPENCV\顶帽和黑帽.png)

```python
img3 = cv.imread("img\\j.png")
cv.imshow("orign3",img3)
kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
gradient = cv.morphologyEx(img3, cv.MORPH_GRADIENT, kernel)
#基本梯度 膨胀-腐蚀，结果就是图像的一个轮廓
cv.imshow("gradient_image",gradient)

em = cv.erode(img3,kernel)
dm = cv.dilate(img3,kernel)
#内部梯度 原图-腐蚀
internal = cv.subtract(img3,em)
#外部梯度 膨胀-原图
external = cv.subtract(dm,img3)
```

![形态学梯度](C:\Users\LEVI\Desktop\OPENCV\形态学梯度.png)

## 15.分水岭算法

图像分割解释：https://people.cmm.minesparis.psl.eu/users/beucher/wtshed.html

参考论文：“Meyer, F. Color Image Segmentation, ICIP92, 1992”

​		任何灰度图像都可以看作是一个地形表面，其中高强度表示山峰，低强度表示山谷。你开始用不同颜色的水(标签)填充每个孤立的山谷(局部最小值)。随着水位的上升，根据附近的山峰(坡度)，来自不同山谷的水明显会开始合并，颜色也不同。为了避免这种情况，你要在水融合的地方建造屏障。你继续填满水，建造障碍，直到所有的山峰都在水下。然后你创建的屏障将返回你的分割结果。这就Watershed背后的“思想”。

​                                            ![lpe1](C:\Users\LEVI\Desktop\OPENCV\lpe1.gif) ![ima3](C:\Users\LEVI\Desktop\OPENCV\ima3.gif)

​		但是这种方法会由于图像中的噪声或其他不规则性而产生过度分割的结果。因此OpenCV实现了一个**基于标记的分水岭算法**，你可以指定哪些是要合并的山谷点，哪些不是。这是一个交互式的图像分割。我们所做的是给我们知道的对象赋予不同的标签。用一种颜色(或强度)标记我们确定为前景或对象的区域，用另一种颜色标记我们确定为背景或非对象的区域，最后用0标记我们不确定的区域。这是我们的标记。然后应用分水岭算法。然后我们的标记将使用我们给出的标签进行更新，对象的边界值将为-1。

![watershed](C:\Users\LEVI\Desktop\OPENCV\watershed.png)

```python
#分水岭算法图像分割过程
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
    

```

![watershed_res](C:\Users\LEVI\Desktop\OPENCV\watershed_res.png)

距离变换算法：cv::distanceTransform

![距离变换](C:\Users\LEVI\Desktop\OPENCV\距离变换.png)

```python
#距离向量
def distanceTransform_demo(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
    kernel = cv.getStructuringElement(cv.MORPH_RECT,(3,3))
    opening = cv.morphologyEx(binary, cv.MORPH_OPEN, kernel, iterations=2)
    res = cv.distanceTransform(opening,cv.DIST_L2,5)
    #归一化
    res_out = cv.normalize(res,0,1.0,cv.NORM_MINMAX)
    cv.imshow("res",res_out*50)
    
distanceTransform(src, distanceType, maskSize, dst=None, dstType=None):
    InputArray src：输入的图像，一般为二值图像
 	OutputArray dst：输出的图像
	int distanceType：所用的求解距离的类型,It can be CV_DIST_L1, CV_DIST_L2 , or CV_DIST_C
	mask_size  距离变换掩模的大小，可以是 3 或 5. 对 CV_DIST_L1 或 CV_DIST_C 的情况，参数值被强制设定为 3, 因为 3×3 mask 给出 5×5 mask 一样的结果，而且速度还更快。 
```

![dist](C:\Users\LEVI\Desktop\OPENCV\dist.png)

## 16.人脸检测

**级联分类器**

​		我们将使用基于Haar Feature的Cascade分类器了解人脸检测和眼睛检测的基础知识。 - 我们将使用**cv::CascadeClassifier**类来检测视频流中的对象。特别是，我们将使用以下函数： - **cv::CascadeClassifier::load**来加载.xml分类器文件。它可以是Haar或LBP分类器 - **cv::CascadeClassifier::detectMultiScale**来执行检测。

​		使用基于Haar特征的级联分类器的对象检测是Paul Viola和Michael Jones在其论文“使用简单特征的增强级联进行快速对象检测”中于2001年提出的一种有效的对象检测方法。这是一种基于机器学习的方法，其中从许多正负图像中训练级联函数。然后用于检测其他图像中的对象。

详细参考---http://www.woshicver.com/Eleventh/10_1_%E7%BA%A7%E8%81%94%E5%88%86%E7%B1%BB%E5%99%A8/

```python
def face_detect_demo(image):
    gray = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
    face_detector = cv.CascadeClassifier("D:\\opencv-4.5.2\\data\\haarcascades\\haarcascade_frontalface_alt_tree.xml")
    #face_detector = cv.CascadeClassifier("D:\\opencv-4.5.2\\data\\lbpcascades\\lbpcascade_frontalface.xml")
    faces = face_detector.detectMultiScale(gray,1.02,5)
    for x,y,w,h in faces:
        cv.rectangle(image,(x,y),(x+w,y+h),(0,0,255),2)
    cv.imshow("result",image)
```

<img src="C:\Users\LEVI\Desktop\OPENCV\face_detetc.png" alt="face_detetc" style="zoom:67%;" />

```python
#视频人脸检测
capture = cv.VideoCapture("img\\Megamind.avi")
while(True):
    ret,frame = capture.read()
    #frame = cv.flip(frame,1)
    face_detect_demo(frame)
    c = cv.waitKey(5)
    if c==27:
        break
```

## 17案例：数字验证码识别

OpenCV+Tesserct-OCR

**Opencv预处理**：去除干扰线与点

Tesserct-OCR验证码识别

​		使用pytesseract时要下载Tesserct-OCR这个软件。根据https://github.com/tesseract-ocr/tesseract/wiki，我找到非官方的安装包，好像我只看到64位的安装包http://digi.bib.uni-mannheim.de/tesseract/tesseract-ocr-setup-4.00.00dev.exe，下载后直接安装即可，但是要记得你的安装目录，我们等会配置环境变量要用。如果不是做英文的图文识别，还需要下载其他语言的识别包https://github.com/tesseract-ocr/tesseract/wiki/Data-Files。简体字识别包：https://raw.githubusercontent.com/tesseract-ocr/tessdata/4.00/chi_sim.traineddata繁体字识别包：https://github.com/tesseract-ocr/tessdata/raw/4.0/chi_tra.traineddata。

```python
使用时报错：没找到tesseract
python pytesseract.pytesseract.TesseractNotFoundError: tesseract is not installed or it's not in your path
解决方案：去C:\Users\LEVI\Anaconda3\envs\cv_env\Lib\site-packages\pytesseract（python环境）里面找到pytesseract.py，里面的tesseract_cmd = 'D:\Program Files (x86)\Tesseract-OCR\\tesseract.exe'设置为自己的tesseract.exe路径。
```

代码：

```python
import cv2 as cv
import numpy as np
import pytesseract as tess
from PIL import Image
#Python图像库PIL(Python Image Library)是python的第三方图像处理库。Image.fromarray将图像转化为数组类型
def recognize_test(image):
    #根据实际情况来预处理验证码
    gray = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
    ret,binary = cv.threshold(gray,0,255,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
    k1 = cv.getStructuringElement(cv.MORPH_RECT,(1,6))
    bin1 = cv.morphologyEx(binary, cv.MORPH_OPEN, k1)
    k2 = cv.getStructuringElement(cv.MORPH_RECT,(5,1))
    bin2 = cv.morphologyEx(bin1,cv.MORPH_OPEN,k2)
    cv.imshow("bin",bin2)

    textImage = Image.fromarray(bin2)
    text = tess.image_to_string(textImage)
    print("验证码:%s"%text)
    
src = cv.imread('img\\yzm2.png')
cv.imshow('image',src)
recognize_test(src)

cv.waitKey(0)
cv.destroyAllWindows()
```

![yzm](C:\Users\LEVI\Desktop\OPENCV\yzm.png)

