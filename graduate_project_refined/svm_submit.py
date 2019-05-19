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

    return training_data, training_label,valid_data, valid_label,test_data, test_label

def rgb2gray(images):
    return np.expand_dims(np.dot(images, [0.2990, 0.5870, 0.1140]), axis=3)


def svm_learning(trainx, trainy,batch_size = 100):
    #define tensorflow session and placeholders
    sess = tf.Session()
    x_data = tf.placeholder(shape=[None, 1024], dtype=tf.float32)
    y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)
    grid = tf.placeholder(shape=[None, 1024], dtype=tf.float32)

    b = tf.Variable(tf.random_normal(shape=[1, batch_size]))

#define rbf kernel for downscale
    gamma = tf.constant(-1.0)#constant for kernel function, minus
    abs_square = tf.multiply(2., tf.matmul(x_data, tf.transpose(x_data)))
    kernel = tf.exp(tf.multiply(gamma, tf.abs(abs_square)))

#compute lagrangian function
    first_sum = tf.reduce_sum(b)
    b_mul= tf.matmul(tf.transpose(b), b)
    y_mul = tf.matmul(y_target, tf.transpose(y_target))
    second_sum = tf.reduce_sum(tf.multiply(kernel, tf.multiply(b_mul, y_mul)))
    loss = tf.negative(tf.subtract(first_sum, second_sum))#minimize the negtive equals to maximize the lagrangian function

#predict the classification result
    rA = tf.reshape(tf.reduce_sum(tf.square(x_data), 1), [-1, 1])
    rB = tf.reshape(tf.reduce_sum(tf.square(grid), 1), [-1, 1])
    pred_sq_dist = tf.add(tf.subtract(rA, tf.multiply(2., tf.matmul(x_data, tf.transpose(grid)))),
                          tf.transpose(rB))
    kernel_prediction = tf.exp(tf.multiply(gamma, tf.abs(pred_sq_dist)))

    prediction_output = tf.matmul(tf.multiply(tf.transpose(y_target), b), kernel_prediction)
    prediction = tf.sign(prediction_output - tf.reduce_mean(prediction_output))
    acc = tf.reduce_mean(tf.cast(tf.equal(tf.squeeze(tf.transpose(prediction)), tf.squeeze(y_target)), tf.float32))
    #optimizer=tf.train.GradientDescentOptimizer(0.001).minimize(loss)
    optimizer = tf.train.AdamOptimizer(0.001).minimize(loss)
    init = tf.global_variables_initializer()
    sess.run(init)
    # Training
    loss_vec = []
    batch_accuracy = []

    for i in range(2000):
        #shuffle data
        rand_index = np.random.choice(len(d1), size=batch_size)
        rand_x = trainx[rand_index]
        rand_y = trainy[rand_index]
        sess.run(optimizer, feed_dict={x_data: rand_x, y_target: rand_y})#run gradient descent optimzier to minimize loss
        #pre=sess.run(prediction, feed_dict={x_data: rand_x, y_target: rand_y,prediction_grid: rand_x})
        #print(pre.shape)
        temp_loss = sess.run(loss, feed_dict={x_data: rand_x, y_target: rand_y})
        loss_vec.append(temp_loss)

        acc_temp = sess.run(acc, feed_dict={x_data: rand_x,
                                                 y_target: rand_y,
                                                 grid: rand_x})
        batch_accuracy.append(acc_temp)
        # print('run:' + str(i + 1))
        # print('Loss:' + str(temp_loss))
        # print('acc:' + str(acc_temp))
        if (i) % 50 == 0:
            print('run:' + str(i + 1))
            print('Loss:' + str(temp_loss))
            print('acc:' + str(acc_temp))
    plt.figure(1)
    plt.ylabel("Loss")
    plt.title('SVM Loss')
    plt.plot(loss_vec, color='RED', label="Train Data")
    plt.legend(loc=2, bbox_to_anchor=(1.05, 1.0), borderaxespad=0.)
    plt.figure(2)
    plt.ylabel("Accuracy")
    plt.title('SVM Accuracy')
    plt.plot(batch_accuracy, color='RED', label="acc")
    plt.legend(loc=2, bbox_to_anchor=(1.05, 1.0), borderaxespad=0.)
    plt.show()
start = time. time()
d1,l1,d2,l2,d3,l3=load_data()
grey_test=np.reshape(rgb2gray(d3),[-1,1024])
grey_valid=np.reshape(rgb2gray(d2),[-1,1024])
grey_train=np.reshape(rgb2gray(d1),[-1,1024])
label_2=np.ones([6000,1])
label_2[l1==0]=-1

train_labels=np.concatenate((l1,label_2),axis=1)
print(label_2[:10])
valid_label_2=np.ones([1000,1])
valid_label_2[l2==1]=0
valid_labels=np.concatenate((l2,valid_label_2),axis=1)

test_label_2=np.ones([int(len(l3)),1])
test_label_2[l3==1]=0
test_labels=np.concatenate((l3,test_label_2),axis=1)
svm_learning(grey_train,label_2)
end = time. time()
print(end - start)