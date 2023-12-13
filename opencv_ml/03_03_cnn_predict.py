import numpy as np
import tensorflow as tf
import cv2 as cv

def normalize_data(data):
    return (data - data.mean()) / data.max()

def load_image(image_path, width, height):
    gray_image = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
    resized_image = cv.resize(gray_image, (width, height))
    normalized_image = normalize_data(resized_image)
    data = []
    data.append(normalized_image.ravel())
    return np.array(data)

### 模型定义开始
IMAGE_WIDTH = 20
IMAGE_HEIGHT = 20
CLASSIFICATION_COUNT = 34
ENGLISH_LABELS = [
	'0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
	'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H',	'J', 'K',
	'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
	'W', 'X', 'Y', 'Z']


def onehot_labels(labels):
    onehots = np.zeros((len(labels), CLASSIFICATION_COUNT))
    for i in np.arange(len(labels)):
        onehots[i, labels[i]] = 1
    return onehots

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    # padding='SAME',使卷积输出的尺寸=ceil(输入尺寸/stride)，必要时自动padding
    # padding='VALID',不会自动padding，对于输入图像右边和下边多余的元素，直接丢弃
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

x = tf.placeholder(tf.float32, shape=[None, IMAGE_HEIGHT * IMAGE_WIDTH])
y_ = tf.placeholder(tf.float32, shape=[None, CLASSIFICATION_COUNT])
x_image = tf.reshape(x, [-1, IMAGE_HEIGHT, IMAGE_WIDTH, 1])

W_conv1 = weight_variable([5, 5, 1, 32])                       # color channel == 1; 32 filters
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)       # 20x20
h_pool1 = max_pool_2x2(h_conv1)                                # 20x20 => 10x10

W_conv2 = weight_variable([5, 5, 32, 64])                      
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)        # 10x10
h_pool2 = max_pool_2x2(h_conv2)                                 # 10x10 => 5x5

# 全连接神经网络的第一个隐藏层
# 池化层输出的元素总数为：5(H)*5(W)*64(filters)
W_fc1 = weight_variable([5 * 5 * 64, 1024])                     # 全连接第一个隐藏层神经元1024个
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 5 * 5 * 64])            # 转成1列
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)      # Affine+ReLU

keep_prob = tf.placeholder(tf.float32)                          # 定义Dropout的比例
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)                    # 执行dropout

# 全连接神经网络输出层
W_fc2 = weight_variable([1024, CLASSIFICATION_COUNT])                             # 全连接输出为10个
b_fc2 = bias_variable([CLASSIFICATION_COUNT])
y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

### 模型定义结束

ENGLISH_MODEL_PATH = "model/cnn_enu/enu.ckpt"
sess = tf.Session()
saver = tf.train.Saver()
saver.restore(sess, ENGLISH_MODEL_PATH)

digit_image_path = "images/digit.jpg"
digit_image = load_image(digit_image_path, IMAGE_WIDTH, IMAGE_HEIGHT)
results = sess.run(y_conv, feed_dict={x: digit_image, keep_prob: 1.0})
predict = np.argmax(results[0])
print(ENGLISH_LABELS[predict])