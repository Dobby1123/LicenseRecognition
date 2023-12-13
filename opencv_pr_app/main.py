import numpy as np
import cv2 as cv
import tensorflow as tf
import sobelplatelocator
import hsvplatelocator
import charseperator
import enucharsrecognizer
import chscharsrecognizer

print("装载模型中......")
chs_model = chscharsrecognizer.load_model()
enu_model = enucharsrecognizer.load_model()
print("装载完成")

print("车牌识别中......")
plate_file_path = "images/plate1.jpg" 
plate_image = cv.imread(plate_file_path)

cv.imshow("", plate_image)
cv.waitKey()

sobel_plates = sobelplatelocator.get_candidate_plates(plate_image)
hsv_plates = hsvplatelocator.get_candidate_plates(plate_image)
candidate_plates = []
for p in sobel_plates:
    candidate_plates.append(p)
for p in hsv_plates:
    candidate_plates.append(p)

for p in candidate_plates:
    cv.imshow("", p)
    cv.waitKey()

for i in np.arange(len(candidate_plates)):
    candidate_plate_image = candidate_plates[i]
    chars_image = charseperator.get_candidate_chars(candidate_plate_image)
    if len(chars_image) < 7 : # 拆分无效
        continue
    chs_char = chscharsrecognizer.predict(chars_image[0], chs_model)
    print(chs_char)
    for j in np.arange(1, len(chars_image)):
        enu_char = enucharsrecognizer.predict(chars_image[j], enu_model)
        print(enu_char)

cv.destroyAllWindows()