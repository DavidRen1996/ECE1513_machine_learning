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

    training_data, training_label = Data[:6000] , Target[:6000]
    valid_data, valid_label = Data[6000:7000], Target[6000:7000]
    test_data, test_label = Data[7000:10000], Target[7000:10000]
    '''training_data, training_label = datax[:10000] / 255, datay[:10000]
    training_label[training_label == 10] = 0
    valid_data, valid_label = datax[10000:16000] / 255, datay[10000:16000]
    valid_label[valid_label == 10] = 0
    test_data, test_label = datax[16000:70000] / 255, datay[16000:70000]
    test_label[test_label == 10] = 0'''
    return training_data, training_label,valid_data, valid_label,test_data, test_label

def rgb2gray(images):
    return np.expand_dims(np.dot(images, [0.2990, 0.5870, 0.1140]), axis=3)
def cnn_learning(train_data,train_label,test_data,test_label,valid_data,valid_label,batch_size,epochs):
    #define placeholders
    x_shaped = tf.placeholder(tf.float32, [None,  32, 32, 1])
    y = tf.placeholder(tf.float32, [None, 2])

#define first conv layer
    conv_filt_shape = [4, 4, 1, 32]
    weights = tf.Variable(tf.truncated_normal(conv_filt_shape,mean=0.0,stddev=0.05))
    bias = tf.Variable(tf.truncated_normal([32],mean=0.0,stddev=0.05))
    out_layer = tf.nn.conv2d(x_shaped, weights, [1, 1, 1, 1], padding='SAME')
    out_layer += bias

#ReLU activate function
    out_layer = tf.nn.relu(out_layer)
#max pooling 2*2
    ksize = [1, 2, 2, 1]
    strides = [1, 2, 2, 1]
    out_layer_1 = tf.nn.max_pool(out_layer, ksize=ksize, strides=strides, padding='SAME')
# define second conv layer
    conv_filt_shape_2 = [4, 4, 32, 64]#filter shape change because of the first conv layer filter changed data shape
    weights_2 = tf.Variable(tf.truncated_normal(conv_filt_shape_2, mean=0.0,stddev=0.05))
    bias_2 = tf.Variable(tf.truncated_normal([64], mean=0.0,stddev=0.05))
    out_layer_2 = tf.nn.conv2d(out_layer_1, weights_2, [1, 1, 1, 1], padding='SAME')
    out_layer_2 += bias_2
    out_layer_2 = tf.nn.relu(out_layer_2)
    ksize = [1, 2, 2, 1]
    strides = [1, 2, 2, 1]
    out_layer_n2 = tf.nn.max_pool(out_layer_2, ksize=ksize, strides=strides, padding='SAME')
    flattened = tf.reshape(out_layer_n2, [-1, 8 * 8 * 64])
#fully connected layer 1
    wd1 = tf.Variable(tf.truncated_normal([8 * 8 * 64, 1024],mean=0.0, stddev=0.03))
    bd1 = tf.Variable(tf.truncated_normal([1024], mean=0.0,stddev=0.01))
    dense_layer1 = tf.matmul(flattened, wd1) + bd1
    dense_layer1 = tf.nn.relu(dense_layer1)
# fully connected layer 2
    wd2 = tf.Variable(tf.truncated_normal([1024, 2], mean=0.0,stddev=0.03), name='wd2')
    bd2 = tf.Variable(tf.truncated_normal([2], mean=0.0,stddev=0.01), name='bd2')
    dense_layer2 = tf.matmul(dense_layer1, wd2) + bd2
    y_ = tf.nn.softmax(dense_layer2)
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=dense_layer2, labels=y))
    optimiser = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    init_op = tf.global_variables_initializer()

    with tf.Session() as sess:
        loss_list = []
        accurate_list = []
        loss_test = []
        loss_valid = []
        acc_test = []
        acc_valid = []
        # initialise the variables
        sess.run(init_op)
        total_batch = int(6000/ batch_size)
        print('total batch:',total_batch)
        for epoch in range(epochs):
            print('epoch:',epoch)
            avg_cost = 0
            #shuffle data set
            randIndex = np.array(range(6000))
            np.random.shuffle(randIndex)
            train_data = train_data[randIndex]

            train_label = train_label[randIndex]
            for i in range(total_batch):
                train_x = train_data[batch_size * i:batch_size * (i + 1)]

                label_y = train_label[batch_size * i: batch_size * (i + 1)]
                #print(out_layer_2.eval(feed_dict={x_shaped:train_x,y:label_y}).shape)

                sess.run(optimiser,feed_dict={x_shaped:train_x,y:label_y})
                if i%5==0:

                    error = sess.run(cross_entropy, feed_dict={x_shaped:train_x,y:label_y})
                    acc=sess.run(accuracy, feed_dict={x_shaped:train_x,y:label_y})
                    #error_valid = sess.run(cross_entropy, feed_dict={x_shaped: valid_data, y: valid_label})
                    #error_test = sess.run(cross_entropy, feed_dict={x_shaped: test_data, y: test_label})
                    print('train error:',error)
                    #print('valid error:', error_valid)
                    #print('test error:', error_test)
                    loss_list.append(error)
                    accurate_list.append(acc)
                    #loss_test.append(error_test)
                    #loss_valid.append(error_valid)

        plt.figure(1)
        plt.ylabel("Loss")
        plt.title('CNN Loss')
        plt.plot(loss_list, color='RED',label="Train Data")
        plt.legend(loc=2, bbox_to_anchor=(1.05, 1.0), borderaxespad=0.)
        plt.figure(2)
        plt.ylabel("Accuracy")
        plt.title('CNN Accuracy')
        plt.plot(accurate_list, color='blue',label="Train Data")
        plt.legend(loc=2, bbox_to_anchor=(1.05, 1.0), borderaxespad=0.)
        #plt.plot(loss_valid, color='green')
        plt.show()

start = time. time()
d1,l1,d2,l2,d3,l3=load_data()
grey_test=rgb2gray(d3)
grey_valid=rgb2gray(d2)
grey_train=rgb2gray(d1)
label_2=np.ones([6000,1])
label_2[l1==1]=0
train_labels=np.concatenate((l1,label_2),axis=1)
print(train_labels[:10])
valid_label_2=np.ones([1000,1])
valid_label_2[l2==1]=0
valid_labels=np.concatenate((l2,valid_label_2),axis=1)

test_label_2=np.ones([int(len(l3)),1])
test_label_2[l3==1]=0
test_labels=np.concatenate((l3,test_label_2),axis=1)
#print(grey_train.shape)
cnn_learning(grey_train,train_labels,grey_test,test_labels,grey_valid,valid_labels,50,5)

end = time. time()
print(end - start)