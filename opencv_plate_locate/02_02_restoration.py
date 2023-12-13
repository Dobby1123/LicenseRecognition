import numpy as np
import cv2 as cv

plate_file_path = "images/restoration1.jpg"
plate_image = cv.imread(plate_file_path)

### 图像预处理
blured_image = cv.GaussianBlur(plate_image, (5, 5), 0)
gray_image = cv.cvtColor(blured_image, cv.COLOR_BGR2GRAY)
grad_x = cv.Sobel(gray_image, cv.CV_16S, 1, 0, ksize=3)
abs_grad_x = cv.convertScaleAbs(grad_x)
grad_image = cv.addWeighted(abs_grad_x, 1, 0, 0, 0)
ret, threshold_image = cv.threshold(grad_image, 0, 255, cv.THRESH_OTSU)
kernel = cv.getStructuringElement(cv.MORPH_RECT, (10, 3))
morphology_image = cv.morphologyEx(threshold_image, cv.MORPH_CLOSE, kernel)
contours, _ = cv.findContours(morphology_image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

### 执行畸变矫正
contour = contours[0]               # 本例中假设有且仅有1个等值线框
rect = cv.minAreaRect(contour)      # rect结构： (center (x,y), (width, h eight), angle of rotation)
rect_width, rect_height = np.int0(rect[1])      # 转成整数
angle = np.abs(rect[2])             # 获得畸变角度
# 如果宽度比高度小，说明矩形相对于最低角点而言，在第二象限；否则相对于最低角点而言在第一象限
if  rect_width < rect_height:
    temp = rect_height
    rect_height = rect_width
    rect_width = temp
    angle = 90 + angle

# 获取该等值线框对应的外接正交矩形(长和宽分别与水平和竖直方向平行)
x, y, w, h = cv.boundingRect(contour)
bounding_image = plate_image[y : y + h, x : x + w]

# 创建一个放大的图像，以便存放之前图像旋转后的结果
enlarged_width = w * 3 // 2
enlarged_height = h * 3 // 2
enlarged_image = np.zeros((enlarged_height, enlarged_width, plate_image.shape[2]), dtype=plate_image.dtype)
x_in_enlarged = (enlarged_width - w) // 2
y_in_enlarged = (enlarged_height - h) // 2
roi_image = enlarged_image[y_in_enlarged:y_in_enlarged+h, x_in_enlarged:x_in_enlarged+w, :]
# 将旋转前的图像拷贝到放大图像的中心位置，注意，为了图像完整性，应拷贝boundingRect中的内容
cv.addWeighted(roi_image, 0, bounding_image, 1, 0, roi_image)
# 计算旋转中心。此处直接使用放大图像的中点作为旋转中心
new_center = (enlarged_width // 2, enlarged_height // 2)
# 获取执行旋转所需的变换矩阵
transform_matrix = cv.getRotationMatrix2D(new_center, -angle, 1.0)   # 角度为正，表明逆时针旋转
# 执行旋转
transformed_image = cv.warpAffine(enlarged_image, transform_matrix, (enlarged_width, enlarged_height))

cv.imshow("", transformed_image)
cv.waitKey()

# 截取与最初等值线框长、宽相同的部分
output_image = cv.getRectSubPix(transformed_image, (rect_width, rect_height), new_center)
cv.imshow("", output_image)
cv.waitKey()
