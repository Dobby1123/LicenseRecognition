import numpy as np
import cv2 as cv

plate_file_path = "images/restoration1.jpg"
plate_image = cv.imread(plate_file_path)

### 图像预处理
# (1)高斯模糊
blured_image = cv.GaussianBlur(plate_image, (5, 5), 0)
# (2)转成灰度图
gray_image = cv.cvtColor(blured_image, cv.COLOR_BGR2GRAY)
# (3)使用Sobel算子，求出水平方向一阶导数，此时应该可以看到较为明显的车牌区域
grad_x = cv.Sobel(gray_image, cv.CV_16S, 1, 0, ksize=3)
abs_grad_x = cv.convertScaleAbs(grad_x)     # 转成CV_8U
# 叠加水平和垂直(本例中没有用到垂直方向)两个方向的梯度结果，形成最终的输出
grad_image = cv.addWeighted(abs_grad_x, 1, 0, 0, 0)
# (4)二值化操作
ret, threshold_image = cv.threshold(grad_image, 0, 255, cv.THRESH_OTSU)
# (5)执行闭操作，使车牌区域连成一个矩形填充的区域
kernel = cv.getStructuringElement(cv.MORPH_RECT, (10, 3))
morphology_image = cv.morphologyEx(threshold_image, cv.MORPH_CLOSE, kernel)
# (6)获取等值线
contours, _ = cv.findContours(morphology_image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
cv.drawContours(plate_image, contours, -1, (0, 0, 255))  

### 查看处于畸变位置的等值线
contour = contours[0]               # 本例中假设有且仅有1个等值线框
rect = cv.minAreaRect(contour)      # rect结构： (center (x,y), (width, height), angle of rotation)
# box是rect四个角点的坐标
box = cv.boxPoints(rect)            
box = np.int0(box)                  # 转成整数
# 画出四个角点
for i in np.arange(len(box)):
    cv.circle(plate_image, tuple(box[i]), 5, (0, 255, 0), 3)

# 获取该等值线框对应的外接正交矩形(长和宽分别与水平和竖直方向平行)
x, y, w, h = cv.boundingRect(contour)
cv.rectangle(plate_image, (x, y), (x+w, y+h), (255, 0, 0), 2)

cv.imshow("", plate_image)
cv.waitKey()
