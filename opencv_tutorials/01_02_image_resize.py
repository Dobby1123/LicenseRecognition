import cv2 as cv

image_path = "images/jf2.jpg"
image_origin = cv.imread(image_path)
# 调整图像大小为480x640，请注意，新尺寸是按照宽x高的顺序来表示
image_resized = cv.resize(image_origin, (480, 640))

cv.imshow("tutorials_origin", image_origin)
cv.imshow("tutorials_resized", image_resized)

cv.waitKey()
cv.destroyAllWindows()

