import os
import numpy as np
import cv2 as cv
from sklearn.neural_network import MLPClassifier
from sklearn.externals import joblib

MODEL_PATH = "model/mlp_chs.m"
TRAIN_DIR = "data/chs_train"
TEST_DIR = "data/chs_test"
IMAGE_WIDTH = 24
IMAGE_HEIGHT = 48
CLASSIFICATION_COUNT = 31
LABEL_DICT = {
	'chuan':0, 'e':1, 'gan':2, 'gan1':3, 'gui':4, 'gui1':5, 'hei':6, 'hu':7, 'ji':8, 'jin':9,
	'jing':10, 'jl':11, 'liao':12, 'lu':13, 'meng':14, 'min':15, 'ning':16, 'qing':17,	'qiong':18, 'shan':19,
	'su':20, 'sx':21, 'wan':22, 'xiang':23, 'xin':24, 'yu':25, 'yu1':26, 'yue':27, 'yun':28, 'zang':29,
	'zhe':30
}


def load_data(dir_path):

    data = []
    labels = []

    for item in os.listdir(dir_path):
        item_path = os.path.join(dir_path, item)
        if os.path.isdir(item_path):
            for subitem in os.listdir(item_path):
                subitem_path = os.path.join(item_path, subitem)
                gray_image = cv.imread(subitem_path, cv.IMREAD_GRAYSCALE)
                resized_image = cv.resize(gray_image, (IMAGE_WIDTH, IMAGE_HEIGHT))
                data.append(resized_image.ravel())
                labels.append(LABEL_DICT[item])
    
    return np.array(data), np.array(labels)

def normalize_data(data):
    return (data - data.mean()) / data.max()

def train():
    print("装载训练数据...")
    train_data, train_labels = load_data(TRAIN_DIR)     
    normalized_data = normalize_data(train_data)
    print("装载%d条数据，每条数据%d个特征" % (normalized_data.shape))   

    print("训练中...")
    model = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(48, 24), random_state=1)
    model.fit(normalized_data, train_labels)
    
    print("训练完成，保存模型...")
    joblib.dump(model, MODEL_PATH)
    print("模型保存到:", MODEL_PATH)


def test():
    print("装载测试数据...")
    test_data, test_labels = load_data(TEST_DIR)     
    normalized_data = normalize_data(test_data)
    print("装载%d条数据，每条数据%d个特征" % (normalized_data.shape)) 

    print("装载模型...")
    model = joblib.load(MODEL_PATH)
    print("模型装载完毕，开始测试...")
    predicts = model.predict(normalized_data)
    errors = np.count_nonzero(predicts - test_labels)
    print("测试完毕，预测正确：%d 条，预测错误:%d 条， 正确率：%f" % 
        (len(predicts) - errors, errors, (len(predicts)-errors) / len(predicts) ))


train()
test()