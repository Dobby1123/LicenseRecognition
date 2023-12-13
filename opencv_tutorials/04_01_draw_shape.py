import cv2 as cv
import numpy as np

WINDOW_TITLE = "Draw Shape"

# 创建480行x640x列x3通道的数组，它实际上代表了高度480，宽度640，像素颜色值初始化为0(黑色)
image = np.zeros((480, 640, 3))
cv.imshow(WINDOW_TITLE, image)
cv.waitKey()

cv.rectangle(image, (20, 20), (120, 220), (0, 0, 255), 3)       # 绘制红色矩形， 线宽度为3
cv.circle(image, (320, 240), 100, (0, 255, 0), 2)               # 绘制绿色圆，半径100，线宽度为2
cv.line(image, (70, 120), (320, 240), (255, 0, 0), 1)           # 绘制蓝色直线，线宽度为1
cv.putText(image, "Hello OpenCV", (320, 50), cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255))     # 输出白色文本

cv.imshow(WINDOW_TITLE, image)
cv.waitKey()
cv.destroyAllWindows()