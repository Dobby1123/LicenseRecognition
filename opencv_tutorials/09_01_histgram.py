import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

WINDOW_TITLE = "Histgram"
IMAGE_PATH = "images/jf1.jpg"

source_image = cv.imread(IMAGE_PATH)
gray_image = cv.cvtColor(source_image, cv.COLOR_BGR2GRAY)

# 计算灰度图的直方数据
channel_index = 0           # 0代表三个颜色通道中的第一个
hist_size = 256
range = [0, 256]
hist = cv.calcHist([gray_image],[channel_index], None, [hist_size], range)
for i in np.arange(hist_size):
    print("灰度值【%d】的像素个数：%d" % (i, hist[i]))

# 手动绘制直方图
PLOT_WIDTH = 512        # 指定绘图坐标的X轴和Y轴长度
PLOT_HEIGHT = 400
# 归一化，hist中每个元素值将介于0~PLOT_HEIGHT之间。实际上求出了每个元素所占的百分比
normalized_hist = hist / np.max(hist)
hist_image = np.zeros((PLOT_HEIGHT, PLOT_WIDTH, 3))
for i in np.arange(hist_size):
    w = 2
    h = PLOT_HEIGHT * (1 - normalized_hist[i])
    x = i * w
    y = PLOT_HEIGHT
    cv.rectangle(hist_image, (x, y), (x + w, h), (0, 0, 255))

cv.imshow(WINDOW_TITLE, hist_image)
cv.waitKey()
cv.destroyAllWindows()

# 直接生成直方图
plt.hist(gray_image.ravel(), 256, range)
plt.show()


