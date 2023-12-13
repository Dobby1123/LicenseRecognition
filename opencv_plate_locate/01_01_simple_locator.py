import numpy as np
import cv2 as cv

WINDOW_TITLE = "Plage Locate"
plate_file_path = "images/京A82806.jpg"

### 步骤1：读取文件，预处理并获取等值线框
# 在包含中文路径时，可采用下列方法读取图像文件
origin_image = cv.imdecode(np.fromfile(plate_file_path, dtype=np.uint8), -1)
# 对原始图片进行高斯模糊
blured_image = cv.GaussianBlur(origin_image, (5, 5), 0)
# 灰度化和二值化处理
gray_image = cv.cvtColor(blured_image, cv.COLOR_BGR2GRAY)
ret, binary_image = cv.threshold(gray_image, 0, 255, cv.THRESH_OTSU)
# 计算等值线矩形
contours, _ = cv.findContours(binary_image, cv.RETR_CCOMP, cv.CHAIN_APPROX_NONE)
# 在原始图片上绘制等值线框
cv.drawContours(origin_image, contours, -1, (0,0,255))
cv.imshow(WINDOW_TITLE, origin_image)
cv.waitKey()

### 步骤2：对等值线框进行筛选，得到候选车牌区域
# 检查每个矩形大小以筛选可能的车牌区域
candidate_regions = []
for i in np.arange(len(contours)):
    x, y, w, h = cv.boundingRect(contours[i])
    ratio = w * 1.0 / h            # 计算区域的宽高比
    if ratio < 1:                  # 支持竖排的情形
        ratio = 1.0 /ratio
    area = w * h
    if area > 136 * 36 and area < 136 * 36 * 10 and ratio > 2.0 and ratio < 5.0:
        candidate_regions.append(origin_image[y:y+h, x:x+w])

if len(candidate_regions) == 0:
    print("没有找到候选的车牌区域！")

# 显示候选车牌区域的图像
for i in np.arange(len(candidate_regions)):
    cv.imshow(WINDOW_TITLE, candidate_regions[i])
    cv.waitKey()