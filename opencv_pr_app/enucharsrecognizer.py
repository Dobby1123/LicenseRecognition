import numpy as np
import cv2 as cv
from sklearn import svm
from sklearn.externals import joblib

IMAGE_WIDTH = 20
IMAGE_HEIGHT = 20
CLASSIFICATION_COUNT = 34
ENGLISH_LABELS = [
	'0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
	'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H',	'J', 'K',
	'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
	'W', 'X', 'Y', 'Z']

def normalize_data(data):
    return (data - data.mean()) / data.max()

def load_model():
    ENGLISH_MODEL_PATH = "model/mlp_enu.m"
    model = joblib.load(ENGLISH_MODEL_PATH)
    return model

def predict(char_image, model):
    # 将图像缩放，以便能放置到(IMAGE_HEIGHT, IMAGE_WIDHT)区域内
    origin_height, origin_width = char_image.shape
    # 如果传入的图片本身就很窄或很矮，不能直接放大图片（例如字符1，直接放大的话就会使1过于膨胀）
    resize_height = IMAGE_HEIGHT-2 if origin_height > IMAGE_HEIGHT else origin_height
    resize_width = IMAGE_WIDTH-2 if origin_width > IMAGE_WIDTH else origin_width
    resized_image = cv.resize(char_image, (resize_width, resize_height))
    
    # 将图像拷贝到工作区的正中间
    working_image = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH))
    x_idx = (IMAGE_WIDTH - resize_width) // 2
    y_idx = (IMAGE_HEIGHT - resize_height) // 2
    working_image[y_idx:y_idx+resize_height, x_idx:x_idx+resize_width] = resized_image

    cv.imshow("", working_image)
    cv.waitKey()

    working_image = normalize_data(working_image)
    data = []
    data.append(working_image.ravel())
    predicts = model.predict(np.array(data))
    return ENGLISH_LABELS[predicts[0]]

