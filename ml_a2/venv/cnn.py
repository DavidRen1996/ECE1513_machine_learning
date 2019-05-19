import numpy as np
import matplotlib.pyplot as plt
import time
import os
import tensorflow as tf
from helper import helpers
helps=helpers()
trainD,validD,testD,trainT,validT,testT=helps.loadData()
new_train,new_valid,new_test=helps.convertOneHot(trainT,validT,testT)

class layers:
    def __init__(self):
        '''self.input_x=tf.placeholder(tf.float32,[sample_length,784], name='input_x')
        self.input_x_shaped=tf.reshape(self.input_x,[-1,28,28,1])
        self.input_y = tf.placeholder(tf.int32, [sample_length,10], name='input_y')
        self.drop_out= tf.placeholder(tf.float32)
        print('initialize layers')'''
        self.input_x = tf.placeholder(tf.float32, name='input_x')
        self.input_x_shaped = tf.reshape(self.input_x, [-1, 28, 28, 1])
        self.input_y = tf.placeholder(tf.int32, name='input_y')

    def conv_network(self,input_data,train_label,test_data,test_label,valid_data,valid_label):
        filter_shape=[3,3,1,32]
        w_con=tf.get_variable("W", shape=filter_shape,initializer=tf.contrib.layers.xavier_initializer())
        b_con = tf.get_variable("b", shape=[32], initializer=tf.contrib.layers.xavier_initializer())
        out_layer = tf.nn.conv2d(self.input_x_shaped, w_con, [1, 1, 1, 1], padding='SAME')+b_con
        relu_layer = tf.nn.relu(out_layer)
        x_mean,x_variance=tf.nn.moments(self.input_x_shaped,axes=[0,1,2])
        batch_normal=tf.nn.batch_normalization(relu_layer,x_mean,x_variance,offset=0,scale=1,variance_epsilon=10**-10)
        #batch_normal=tf.nn.batch_normalization(relu_layer,0,1,offset=0.01,scale=None,variance_epsilon=10**-10)
        #batch_normal = tf.layers.batch_normalization(self.input_x, training=True)
        ksize = [1, 2, 2, 1]
        strides = [1, 2, 2, 1]
        out_layer_1 = tf.nn.max_pool(batch_normal, ksize=ksize, strides=strides,padding='SAME')
        flattened = tf.reshape(out_layer_1, [-1, 14 * 14 * 32])
        wd1 = tf.get_variable("wd", shape=[14 * 14 * 32,784],initializer=tf.contrib.layers.xavier_initializer())
        bd1 = tf.get_variable("bd1", shape=[784],initializer=tf.contrib.layers.xavier_initializer())
        dense_layer1 = tf.matmul(flattened, wd1) + bd1
        relu_layer1 = tf.nn.relu(dense_layer1)
        wd2 = tf.get_variable("wd2", shape=[784,10],initializer=tf.contrib.layers.xavier_initializer())
        bd2 = tf.get_variable("bd2", shape=[10],initializer=tf.contrib.layers.xavier_initializer())
        dense_layer2 = tf.matmul(relu_layer1, wd2) + bd2
        y_hat = tf.nn.softmax(dense_layer2)
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=dense_layer2, labels=self.input_y))
        optimizer = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(cross_entropy)
        correct_prediction = tf.equal(tf.argmax(self.input_y, 1), tf.argmax(y_hat, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        init = tf.global_variables_initializer()

        with tf.Session() as sess:
            sess.run(init)
            total_batch=round(10000/32)#312
            loss_list=[]
            accurate_list=[]
            loss_test=[]
            loss_valid=[]
            acc_test=[]
            acc_valid=[]
            for i in range(50):

                randIndex=np.array(range(10000))
                np.random.shuffle(randIndex)
                input_data=input_data[randIndex]

                train_label = train_label[randIndex]
                print(i)
                for batch in range(total_batch):

                    train_x=input_data[32*batch:32*(batch+1)]

                    label_y = train_label[32 * batch: 32 * (batch + 1)]
                    feed_dict={self.input_x: train_x, self.input_y: label_y}
                    sess.run(optimizer,feed_dict)

                    if batch==total_batch-1:
                        error = sess.run(cross_entropy, feed_dict)
                        error_test=sess.run(cross_entropy,feed_dict={self.input_x: test_data, self.input_y: test_label})
                        error_valid=sess.run(cross_entropy,feed_dict={self.input_x: valid_data, self.input_y: valid_label})
                        loss_test.append(error_test)
                        loss_valid.append(error_valid)

                        print(error)
                        loss_list.append(error)
                        acc = sess.run(accuracy, feed_dict)
                        acc_t=sess.run(accuracy, feed_dict={self.input_x: test_data, self.input_y: test_label})
                        acc_v=sess.run(accuracy, feed_dict={self.input_x: valid_data, self.input_y: valid_label})
                        accurate_list.append(acc)
                        acc_test.append(acc_t)
                        acc_valid.append(acc_v)
                        print('train accuracy:',acc)
                        print('test accuracy:',acc_t)
                        print('valid accuracy:',acc_v)
                        #error = sess.run(cross_entropy, feed_dict)
                        #print(error)


            print('plot figure')
            plt.figure(1)
            plt.plot(loss_list, color='RED')
            plt.plot(loss_test)
            plt.plot(loss_valid, color='green')
            plt.figure(2)
            plt.plot(accurate_list, color='RED')
            plt.plot(acc_test)
            plt.plot(acc_valid, color='green')
            plt.show()
            #feed_dict = {self.input_x: input_data, self.input_y: train_label}
            #error=sess.run(cross_entropy, feed_dict)
            #print(error)


cnn_layer=layers()
reshaped_trainD=helps.reshape_datamatrix(trainD)
reshaped_test=helps.reshape_datamatrix(testD)
reshaped_valid=helps.reshape_datamatrix(validD)
cnn_layer.conv_network(reshaped_trainD,new_train,reshaped_test,new_test,reshaped_valid,new_valid)