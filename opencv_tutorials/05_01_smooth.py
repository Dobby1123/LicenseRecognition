import cv2 as cv

WINDOW_TITLE = "Image Smooth"
IMAGE_PATH = "images/chess.jpg"
KERNEL_SIZE_X = 11
KERNEL_SIZE_Y = 11

src = cv.imread(IMAGE_PATH)

# 均值平滑
tar1 = cv.blur(src, (KERNEL_SIZE_X, KERNEL_SIZE_Y))

# 高斯平滑, 核尺寸必须为奇数
tar2 = cv.GaussianBlur(src, (KERNEL_SIZE_X, KERNEL_SIZE_Y), 0)

cv.imshow(WINDOW_TITLE, src)
cv.waitKey()
cv.imshow(WINDOW_TITLE, tar1)
cv.waitKey()
cv.imshow(WINDOW_TITLE, tar2)
cv.waitKey()
cv.destroyAllWindows()