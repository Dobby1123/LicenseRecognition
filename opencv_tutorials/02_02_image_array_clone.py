import numpy as np
import cv2 as cv

image_path = "images/jf1.jpg"
image1 = cv.imread(image_path)

window_name = "tutorials"
cv.namedWindow(window_name)

# 浅层复制，修改image2中的数据，相当于也修改了image1中的数据
image2 = image1
# 此时二者维度、元素值完全相同，且共享同一块图像元素内存
print("image1的维度：", image1.shape, "；image2的维度：", image2.shape)
# 将image2中所有数据全设为0
image2[:,:,:] = 0
# image1此时也全都变为黑色(颜色值为0)
cv.imshow(window_name, image1)
cv.waitKey()

# 重新装载image1以进行后续操作
image1 = cv.imread(image_path)

# 深层复制，修改image3中的数据，对image1没有影响
image3 = np.copy(image1)
# 此时二者维度、元素值完全相同，但分别具有各自的图像元素内存
print("image1的维度：", image1.shape, "；image3的维度：", image3.shape)
# 将image3中所有数据全设为0
image3[:,:,:] = 0
# image1完全不受影响
cv.imshow(window_name, image1)

cv.waitKey()
cv.destroyAllWindows()