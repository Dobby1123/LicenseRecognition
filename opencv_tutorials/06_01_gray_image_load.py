import cv2 as cv

WINDOW_TITLE = "Gray Image"
IMAGE_PATH = "images/chess.jpg"

# 直接以灰度图的形式读取元图片
gray_image1 = cv.imread(IMAGE_PATH, cv.IMREAD_GRAYSCALE)

# 读取原图片，然后转成灰度图
source_image = cv.imread(IMAGE_PATH)
gray_image2 = cv.cvtColor(source_image, cv.COLOR_BGR2GRAY)

cv.imshow(WINDOW_TITLE, gray_image1)
cv.waitKey()
cv.imshow(WINDOW_TITLE, gray_image2)
cv.waitKey()
cv.destroyAllWindows()