import numpy as np
import tensorflow as tf
import cv2 as cv

IMAGE_WIDTH = 24
IMAGE_HEIGHT = 48
CLASSIFICATION_COUNT = 31
CHINESE_LABELS = [
	"川","鄂","赣","甘","贵","桂","黑","沪","冀","津",
	"京","吉","辽","鲁","蒙","闽","宁","青","琼","陕",
	"苏","晋","皖","湘","新","豫","渝","粤","云","藏",
	"浙"]

def normalize_data(data):
    return (data - data.mean()) / data.max()

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


def load_model():
    x = tf.placeholder(tf.float32, shape=[None, IMAGE_HEIGHT * IMAGE_WIDTH])
    y_ = tf.placeholder(tf.float32, shape=[None, CLASSIFICATION_COUNT])
    x_image = tf.reshape(x, [-1, IMAGE_HEIGHT, IMAGE_WIDTH, 1])

    W_conv1 = weight_variable([5, 5, 1, 32])                       # color channel == 1; 32 filters
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)       # 24x48
    h_pool1 = max_pool_2x2(h_conv1)                                # 24x48 => 12x24

    W_conv2 = weight_variable([5, 5, 32, 64])                      
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)        # 12x24
    h_pool2 = max_pool_2x2(h_conv2)                                 # 12x24 => 6x12

    # 全连接神经网络的第一个隐藏层
    # 池化层输出的元素总数为：12(H)*24(W)*64(filters)
    W_fc1 = weight_variable([6 * 12 * 64, 1024])                     # 全连接第一个隐藏层神经元1024个
    b_fc1 = bias_variable([1024])
    h_pool2_flat = tf.reshape(h_pool2, [-1, 6 * 12 * 64])            # 转成1列
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)      # Affine+ReLU

    keep_prob = tf.placeholder(tf.float32)                          # 定义Dropout的比例
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)                    # 执行dropout

    # 全连接神经网络输出层
    W_fc2 = weight_variable([1024, CLASSIFICATION_COUNT])                             # 全连接输出为10个
    b_fc2 = bias_variable([CLASSIFICATION_COUNT])
    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    learning_rate = 1e-4
    max_epochs = 20
    batch_size = 50
    check_step = 50

    # 使用softmax成本函数
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    MODEL_PATH = "model/cnn_chs/chs.ckpt"
    sess = tf.Session()
    saver = tf.train.Saver()
    saver.restore(sess, MODEL_PATH)

    return (sess, x, keep_prob, y_conv)

def predict(char_image, model):
    digit_images = []
    working_image = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH))
    resized_image = cv.resize(char_image, (IMAGE_WIDTH-4, IMAGE_HEIGHT-2))
    working_image[1:-1, 2:-2] = resized_image
    working_image = normalize_data(working_image)
    digit_images.append(working_image.ravel())
    sess, x, keep_prob, y_conv = model
    results = sess.run(y_conv, feed_dict={x: digit_images, keep_prob: 1.0})
    predict = np.argmax(results[0])
    return CHINESE_LABELS[predict]

