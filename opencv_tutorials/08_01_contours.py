import cv2 as cv

WINDOW_TITLE = "Contours"
IMAGE_PATH = "images/contours.jpg"

source_image = cv.imread(IMAGE_PATH)
gray_image = cv.cvtColor(source_image, cv.COLOR_BGR2GRAY)
ret, binary_image = cv.threshold(gray_image, 0, 255, cv.THRESH_OTSU)
contours, _ = cv.findContours(binary_image, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
cv.drawContours(source_image, contours, -1, (0, 0, 255))        # 以红色绘制等值线

cv.imshow(WINDOW_TITLE, source_image)
cv.waitKey()
cv.destroyAllWindows()