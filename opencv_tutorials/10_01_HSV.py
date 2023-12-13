import cv2 as cv
import numpy as np

WINDOW_TITLE = "HSV Color Space"
IMAGE_PATH = "images/hsv1.jpg"

source_image = cv.imread(IMAGE_PATH)

# 从BGR颜色空间转变成HSV颜色空间
# 将H,S,V分量分别放到三个array中。其中H分量代表色调；S分量代表饱和度；V分量代表亮度
# H的值本应在0~360之间，但是OpenCV为了保证用UCHAR来表达，所以对该值除以2，因此H的值域是0~180
# S和V分量本应在0~1之间，但OpenCV乘以了255，使其值域是0~255
hsv_image = cv.cvtColor(source_image, cv.COLOR_BGR2HSV)

# 查看若干像素的HSV分量值
for i in np.arange(10, hsv_image.shape[0], 100):
    for j in np.arange(10, hsv_image.shape[1], 100):
        item = hsv_image[i, j]
        h, s, v = item
        print("第【%d, %d】元素，H=%d, S=%d, V=%d" % (i, j, h, s, v))

# 将HSV分拆到三个array中
hsv_splits = cv.split(hsv_image)
h_split = hsv_splits[0]             # 取得H分量数组
# 蓝色色调的范围一般在100~140之间
HSV_MIN_BLUE_H = 100
HSV_MAX_BLUE_H = 140
for i in np.arange(h_split.shape[0]):
    for j in np.arange(h_split.shape[1]):
        h_value = h_split[i, j]
        if (h_value > HSV_MIN_BLUE_H and h_value <= HSV_MAX_BLUE_H):    # 如果该像素是在蓝色色调范围内
            source_image[i, j, :] = 0               # 将原图中对应像素的三个颜色分量全部设置为黑色
        else:
            source_image[i, j, :] = 255             # 置为白色

cv.imshow(WINDOW_TITLE, source_image)
cv.waitKey()
cv.destroyAllWindows()