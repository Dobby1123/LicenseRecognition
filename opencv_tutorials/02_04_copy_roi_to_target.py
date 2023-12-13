import cv2 as cv

image1_path = "images/jf1.jpg"
image2_path = "images/jf2.jpg"
image1 = cv.imread(image1_path)
image2 = cv.imread(image2_path)

# 指定images中要复制的感兴趣区域
roi = image1[100:, 100:, :]
# 注意，roi数组中的第一个维度代表高度(数组行数)
roi_height = roi.shape[0]
# roi数组中的第二个维度代表宽度(数组烈属)
roi_width = roi.shape[1]
# 将roi数据复制到image2的目标区域中
image2[0:roi_height, 0:roi_width, :] = roi

cv.imshow("image2 with roi copied", image2)
cv.waitKey()
cv.destroyAllWindows()