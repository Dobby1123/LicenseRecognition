import numpy as np
import cv2 as cv 
import util

plate_file_path = "images/plate4.jpg"
plate_image = cv.imread(plate_file_path)

### 将输入图片进行预处理，以供后续的等值线生成
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
    
### 执行筛选、矫正和尺寸调整
contours, _ = cv.findContours(morphology_image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
verified_plates = []
for i in np.arange(len(contours)):
    if  util.verify_plate_sizes(contours[i]):
        output_image = util.rotate_plate_image(contours[i], plate_image)
        output_image = util.unify_plate_image(output_image)
        verified_plates.append(output_image)

for i in np.arange(len(verified_plates)):
    cv.imshow("", verified_plates[i])
    cv.waitKey()

cv.destroyAllWindows()