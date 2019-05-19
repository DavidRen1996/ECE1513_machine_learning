import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.io import loadmat


def load_data():
    data = loadmat('E:\\download_backup\\train_32x32.mat')
    datax, datay = data['X'].transpose((3, 0, 1, 2)), data['y'][:, 0]  # shape of (73257, 32, 32, 3) and (73257,)
    np.random.seed(421)
    randIndx = np.arange(73257)
    # print (len(Data))
    # print (randIndx)
    np.random.shuffle(randIndx)
    datax, datay = datax[randIndx], datay[randIndx]
    training_data, training_label = datax[:5000] / 255, datay[:5000]
    print(training_label[10:20, ])
    training_label[training_label == 10] = 0
    valid_data, valid_label = datax[5000:8000] / 255, datay[5000:8000]
    valid_label[valid_label == 10] = 0
    test_data, test_label = datax[8000:25000] / 255, datay[8000:25000]
    test_label[test_label == 10] = 0
    return training_data, training_label, valid_data, valid_label, test_data, test_label


def data_format_train(x, y):
    linear_x = np.reshape(x, [x.shape[0], -1])  # shape of (73257, 3072)
    linear_y = np.reshape(y, [y.shape[0], -1])  # shape of (73257, 1)
    print(linear_y[0])
    return linear_x, linear_y


def binary_labels(training_label, valid_label, test_label):
    hottrain = training_label
    hottrain[training_label != 1] = 0
    hotvalid = valid_label
    hotvalid[valid_label != 1] = 0
    hottest = test_label
    hottest[test_label != 1] = 0

    return hottrain, hotvalid, hottest


def rgb2gray(images):
    return np.expand_dims(np.dot(images, [0.2990, 0.5870, 0.1140]), axis=3)


def learning(trainx, trainy, validx, validy, testx, testy, epoch, batch_number):
    X = tf.placeholder(tf.float32, name="X")
    y = tf.placeholder(tf.float32, name="y")
    W = tf.Variable(tf.truncated_normal(shape=[trainx.shape[1], 1], stddev=0.1), name="weight")
    b = tf.Variable(tf.constant(0.5, shape=[1]))
    y_hat = tf.add(tf.matmul(X, W), b)
    loss = tf.reduce_mean(tf.square(y_hat - y)) / 2 + 0.01 * tf.nn.l2_loss(W)
    # loss=tf.losses.mean_squared_error(y,y_hat)
    # optimizer = tf.train.GradientDescentOptimizer(0.0001).minimize(loss, name='GradientDescentOptimizer')
    optimizer = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(loss)
    init = tf.global_variables_initializer()
    acc_train_list = []
    acc_test_list = []
    acc_valid_list = []
    loss_train_list = []
    loss_test_list = []
    loss_valid_list = []
    with tf.Session() as sess:
        sess.run(init)
        batch_size = int(len(trainx) / batch_number)
        for index in range(epoch):

            print(index)
            for batch_index in range(batch_number):
                # print(batch_size)
                batch_X = trainx[batch_index * batch_size:(batch_index + 1) * batch_size]
                # print(batch_X.shape)
                batch_y = trainy[batch_index * batch_size:(batch_index + 1) * batch_size]
                sess.run(optimizer, feed_dict={X: batch_X, y: batch_y})
                if index % 100 == 0 and batch_index == 1 and index != 0:
                    loss_train = sess.run(loss, feed_dict={X: trainx, y: trainy})
                    loss_test = sess.run(loss, feed_dict={X: testx, y: testy})
                    loss_valid = sess.run(loss, feed_dict={X: validx, y: validy})
                    loss_train_list.append(loss_train)
                    loss_test_list.append(loss_test)
                    loss_valid_list.append(loss_valid)
                    if index == epoch - 50:
                        print('train error:', loss_train)
                        print('test error:', loss_test)
                        print('valid error:', loss_valid)
                    # print('train error:',loss_train)
                    # print('test error:',loss_test)
                    # print('valid error:',loss_valid)
                    acc_train = np.mean((y_hat.eval(feed_dict={X: trainx, y: trainy}) > 0.5) == trainy)
                    acc_valid = np.mean((y_hat.eval(feed_dict={X: validx, y: validy}) > 0.5) == validy)
                    acc_test = np.mean((y_hat.eval(feed_dict={X: testx, y: testy}) > 0.5) == testy)
                    if index == (epoch - 100):
                        print('train acc:', acc_train)
                        print('test acc:', acc_valid)
                        print('valid acc:', acc_test)
                    acc_train_list.append(acc_train)
                    acc_test_list.append(acc_test)
                    acc_valid_list.append(acc_valid)

            randIndex = np.array(range(5000))
            np.random.shuffle(randIndex)
            trainx, trainy = trainx[randIndex], trainy[randIndex]

        plt.figure(1)
        print('plot')
        plt.ylabel("Loss")
        plt.title('Linear Regression Loss')
        plt.plot(loss_train_list, color='RED', label="Train Data")
        plt.plot(loss_test_list, color='green', label="Test Data")
        plt.plot(loss_valid_list, color='blue', label="Valid Data")
        plt.legend(loc=2, bbox_to_anchor=(1.05, 1.0), borderaxespad=0.)

        plt.figure(2)
        print('plot')
        plt.ylabel("Accuracy")
        plt.title('Linear Regression Accuracy')
        plt.plot(acc_train_list, color='RED', label="Train Data")
        plt.plot(acc_test_list, color='green', label="Test Data")
        plt.plot(acc_valid_list, color='blue', label="Valid Data")
        plt.legend(loc=2, bbox_to_anchor=(1.05, 1.0), borderaxespad=0.)

        plt.show()


d1, l1, d2, l2, d3, l3 = load_data()

train_greyscale = rgb2gray(d1).astype(np.float32)
test_greyscale = rgb2gray(d3).astype(np.float32)
val_greyscale = rgb2gray(d2).astype(np.float32)

reshaped_trainx, reshaped_trainy = data_format_train(train_greyscale, l1)
reshaped_testx, reshaped_testy = data_format_train(test_greyscale, l3)
reshaped_validx, reshaped_validy = data_format_train(val_greyscale, l2)
one_train, one_valid, one_test = binary_labels(reshaped_trainy, reshaped_validy, reshaped_testy)
learning(reshaped_trainx, one_train, reshaped_validx, one_valid, reshaped_testx, one_test, 2000, 2)