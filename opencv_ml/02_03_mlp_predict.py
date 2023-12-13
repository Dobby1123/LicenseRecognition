import os
import numpy as np
import cv2 as cv
from sklearn import svm
from sklearn.externals import joblib

def normalize_data(data):
    return (data - data.mean()) / data.max()

def load_image(image_path, width, height):
    gray_image = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
    resized_image = cv.resize(gray_image, (width, height))
    normalized_image = normalize_data(resized_image)
    data = []
    data.append(normalized_image.ravel())
    return np.array(data)

ENGLISH_LABELS = [
	'0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
	'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H',	'J', 'K',
	'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
	'W', 'X', 'Y', 'Z']

CHINESE_LABELS = [
	"川","鄂","赣","甘","贵","桂","黑","沪","冀","津",
	"京","吉","辽","鲁","蒙","闽","宁","青","琼","陕",
	"苏","晋","皖","湘","新","豫","渝","粤","云","藏",
	"浙"]

ENGLISH_MODEL_PATH = "model/mlp_enu.m"
CHINESE_MODEL_PATH = "model/mlp_chs.m"
ENGLISH_IMAGE_WIDTH = 20
ENGLISH_IMAGE_HEIGHT = 20
CHINESE_IMAGE_WIDTH = 24
CHINESE_IMAGE_HEIGHT = 48

digit_image_path = "images/digit.jpg"
english_image_path = "images/english.jpg"
chinese_image_path = "images/chinese.jpg"

# 装载图片、转换成适当的尺寸并作归一化处理
digit_image = load_image(digit_image_path, ENGLISH_IMAGE_WIDTH, ENGLISH_IMAGE_HEIGHT)
english_image = load_image(english_image_path, ENGLISH_IMAGE_WIDTH, ENGLISH_IMAGE_HEIGHT)
chinese_image = load_image(chinese_image_path, CHINESE_IMAGE_WIDTH, CHINESE_IMAGE_HEIGHT)

# 装载模型
enu_model = joblib.load(ENGLISH_MODEL_PATH)
chs_model = joblib.load(CHINESE_MODEL_PATH)

# 执行预测并输出预测的文本结果
predicts = enu_model.predict(digit_image)
print(ENGLISH_LABELS[predicts[0]])
predicts = enu_model.predict(english_image)
print(ENGLISH_LABELS[predicts[0]])
predicts = chs_model.predict(chinese_image)
print(CHINESE_LABELS[predicts[0]])
