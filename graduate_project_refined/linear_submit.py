import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.io import loadmat


def load_data():
    data = loadmat('E:\\download_backup\\train_32x32.mat')
    datax, datay = data['X'].transpose((3, 0, 1, 2)), data['y'][:,0]  # shape of (73257, 32, 32, 3) and (73257,)
    pos_class=1
    neg_class=2
    dataIndx = (datay == pos_class) + (datay == neg_class)
    Data = datax[dataIndx] / 255.
    Target = datay[dataIndx].reshape(-1, 1)

    Target[Target == pos_class] = 1
    Target[Target == neg_class] = 0
    np.random.seed(421)
    randIndx = np.arange(len(Data))

    np.random.shuffle(randIndx)
    Data, Target = Data[randIndx], Target[randIndx]
    print(Data.shape)
    training_data, training_label = Data[:6000] , Target[:6000]
    valid_data, valid_label = Data[6000:7000], Target[6000:7000]
    test_data, test_label = Data[7000:], Target[7000:]
    '''training_data, training_label = datax[:10000] / 255, datay[:10000]
    training_label[training_label == 10] = 0
    valid_data, valid_label = datax[10000:16000] / 255, datay[10000:16000]
    valid_label[valid_label == 10] = 0
    test_data, test_label = datax[16000:70000] / 255, datay[16000:70000]
    test_label[test_label == 10] = 0'''
    return training_data, training_label,valid_data, valid_label,test_data, test_label

def rgb2gray(images):
    return np.expand_dims(np.dot(images, [0.2990, 0.5870, 0.1140]), axis=3)


def learning(trainx, trainy, validx, validy, testx, testy, epoch, batch_number):
    X = tf.placeholder(tf.float32, name="X")
    y = tf.placeholder(tf.float32, name="y")
    W = tf.Variable(tf.truncated_normal(shape=[trainx.shape[1], 2], stddev=0.1), name="weight")
    b = tf.Variable(tf.constant(0.5, shape=[2]))
    y_hat = tf.add(tf.matmul(X, W), b)
    # loss = tf.reduce_mean(tf.square(y_hat - y)) / 2 + 0.01 * tf.nn.l2_loss(W)
    loss = tf.losses.mean_squared_error(y, y_hat) + 0.01 * tf.nn.l2_loss(W)
    # optimizer = tf.train.GradientDescentOptimizer(0.001).minimize(loss, name='GradientDescentOptimizer')
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
                if index % 50 == 0 and batch_index == 1 and index != 0:
                    loss_train = sess.run(loss, feed_dict={X: trainx, y: trainy})
                    loss_test = sess.run(loss, feed_dict={X: testx, y: testy})
                    loss_valid = sess.run(loss, feed_dict={X: validx, y: validy})
                    loss_train_list.append(loss_train)
                    loss_test_list.append(loss_test)
                    loss_valid_list.append(loss_valid)
                    '''
                    if index == epoch - 10:
                        print('train error:', loss_train)
                        print('test error:', loss_test)
                        print('valid error:', loss_valid)
                    '''
                    print('train error:',loss_train)
                    print('test error:',loss_test)
                    print('valid error:',loss_valid)
                    acc_train = np.mean((y_hat.eval(feed_dict={X: trainx, y: trainy}) > 0.5) == trainy)
                    acc_valid = np.mean((y_hat.eval(feed_dict={X: validx, y: validy}) > 0.5) == validy)
                    acc_test = np.mean((y_hat.eval(feed_dict={X: testx, y: testy}) > 0.5) == testy)

                    acc_train_list.append(acc_train)
                    acc_test_list.append(acc_test)
                    acc_valid_list.append(acc_valid)
            #shuffle data
            randIndex = np.array(range(6000))
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


start = time. time()
d1,l1,d2,l2,d3,l3=load_data()
grey_test=np.reshape(rgb2gray(d3),[-1,1024])
grey_valid=np.reshape(rgb2gray(d2),[-1,1024])
grey_train=np.reshape(rgb2gray(d1),[-1,1024])
label_2=np.ones([6000,1])
label_2[l1==1]=0
train_labels=np.concatenate((l1,label_2),axis=1)

valid_label_2=np.ones([1000,1])
valid_label_2[l2==1]=0
valid_labels=np.concatenate((l2,valid_label_2),axis=1)

test_label_2=np.ones([int(len(l3)),1])
test_label_2[l3==1]=0
test_labels=np.concatenate((l3,test_label_2),axis=1)

print(train_labels[:10])
learning(grey_train, train_labels, grey_valid, valid_labels, grey_test, test_labels, 800, 20)

end = time. time()
print(end - start)

