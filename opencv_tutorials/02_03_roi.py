import numpy as np
import cv2 as cv

image_path = "images/jf1.jpg"
image = cv.imread(image_path)

window_name = "tutorials"
cv.namedWindow(window_name)

# 获取图像中的一个矩形子区域，实际上做了一个浅层复制
'''
    请注意，多维数组各个维度顺序为高x宽x通道(HxWxC)
    通过下标索引取数组中的元素时：
        第1个下标代表图片的高度方向，同时也代表数组的行标号
        第2个下标代表图片的宽度方向，同时也代表数组的列标号
        第3个下标代表图片的颜色通道
'''
roi = image[:400, 0:200, :]
cv.imshow(window_name, roi)
cv.waitKey()

# 如果将roi中的颜色值都修改为0，则image图片也会被修改
roi[:,:,:] = 0
cv.imshow(window_name, image)

cv.waitKey()
cv.destroyAllWindows()