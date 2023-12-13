import cv2 as cv
import numpy as np

WINDOW_TITLE = "Morphology"
IMAGE_PATH = "images/morphology.jpg"

source_image = cv.imread(IMAGE_PATH)
gray_image = cv.cvtColor(source_image, cv.COLOR_BGR2GRAY)

ret, binary_image = cv.threshold(gray_image, 0, 255, cv.THRESH_OTSU)

# 执行闭操作
kernel = cv.getStructuringElement(cv.MORPH_RECT, (32, 5))        #尝试调整Size中的核大小，可以得到不同的连接结果
# 闭操作之后，车牌字符区域连接成一片，一般有利于使用等值线轮廓识别该区域矩形
morphology_image = cv.morphologyEx(binary_image, cv.MORPH_CLOSE, kernel)

cv.imshow(WINDOW_TITLE, source_image)
cv.waitKey()
cv.imshow(WINDOW_TITLE, morphology_image)
cv.waitKey()
cv.destroyAllWindows()