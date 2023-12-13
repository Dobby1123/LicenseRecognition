import cv2 as cv

WINDOW_TITLE = "Image Binarize"
IMAGE_PATH = "images/chess.jpg"

# 直接以灰度图的形式读取元图片
gray_image = cv.imread(IMAGE_PATH, cv.IMREAD_GRAYSCALE)

# 指定阈值的二值化处理
threshold_value = 127.0
max_value = 255
ret, binary_image1 = cv.threshold(gray_image, threshold_value, max_value, cv.THRESH_BINARY)
ret, binary_image2 = cv.threshold(gray_image, threshold_value, max_value, cv.THRESH_BINARY_INV)

# 采用最大类间方差/大津法自动优化阈值。
ret, binary_image3 = cv.threshold(gray_image, 0, max_value, cv.THRESH_OTSU)

cv.imshow(WINDOW_TITLE, binary_image1)
cv.waitKey()
cv.imshow(WINDOW_TITLE, binary_image2)
cv.waitKey()
cv.imshow(WINDOW_TITLE, binary_image3)
cv.waitKey()
cv.destroyAllWindows()