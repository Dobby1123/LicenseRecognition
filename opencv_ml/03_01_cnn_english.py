''' 
使用简单卷积神经网络分类，网络结构为：
Conv -> ReLU -> Max Pooling -> Conv -> ReLU -> Max Pooling -> 
FC1(1024) -> ReLU -> Dropout -> Affine -> Softmax
'''

import os
import numpy as np
import tensorflow as tf
import cv2 as cv

MODEL_PATH = "model/cnn_enu/enu.ckpt"
TRAIN_DIR = "data/enu_train"
TEST_DIR = "data/enu_test"
IMAGE_WIDTH = 20
IMAGE_HEIGHT = 20
CLASSIFICATION_COUNT = 34
LABEL_DICT = {
	'0':0, '1':1, '2':2, '3':3, '4':4, '5':5, '6':6, '7':7, '8':8, '9':9,
	'A':10, 'B':11, 'C':12, 'D':13, 'E':14, 'F':15, 'G':16, 'H':17,	'J':18, 'K':19,
	'L':20, 'M':21, 'N':22, 'P':23, 'Q':24, 'R':25, 'S':26, 'T':27, 'U':28, 'V':29,
	'W':30, 'X':31, 'Y':32, 'Z':33
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

learning_rate = 1e-4
max_epochs = 40
batch_size = 50
check_step = 10

# 使用softmax成本函数
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    print("装载训练数据...")
    train_data, train_labels = load_data(TRAIN_DIR)     
    train_data = normalize_data(train_data)
    train_labels = onehot_labels(train_labels)
    print("装载%d条数据，每条数据%d个特征" % (train_data.shape))   
    
    train_samples_count = len(train_data)
    train_indicies = np.arange(train_samples_count)
    # 获得打乱的索引序列
    np.random.shuffle(train_indicies)

    print("装载测试数据...")
    test_data, test_labels = load_data(TEST_DIR)     
    test_data = normalize_data(test_data)
    test_labels = onehot_labels(test_labels)
    print("装载%d条数据，每条数据%d个特征" % (test_data.shape)) 

    iters = int(np.ceil(train_samples_count / batch_size))
    print("Training...")
    for epoch in range(1, max_epochs + 1):
        print("Epoch #", epoch)
        for i in range(1, iters + 1):
            # 获取本批数据
            start_idx = (i * batch_size) % train_samples_count
            idx = train_indicies[start_idx : start_idx + batch_size]
            batch_x = train_data[idx, :]
            batch_y = train_labels[idx, :] 
            _, batch_accuracy = sess.run([train_step, accuracy], feed_dict = { x:batch_x, y_:batch_y, keep_prob:0.5 })
            if i % check_step == 0:
                print("Iter:", i, "of", iters, "batch_accuracy=", batch_accuracy)
    print("Training completed.")

    print("Saving model...")
    saver = tf.train.Saver()
    saved_file = saver.save(sess, MODEL_PATH)
    print('Model saved to ', saved_file)

    test_accuracy = accuracy.eval(feed_dict={x: test_data, y_: test_labels, keep_prob: 1.0})
    print('Test accuracy %g' % test_accuracy )   # 约0.97~0.98