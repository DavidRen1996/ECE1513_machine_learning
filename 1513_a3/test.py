import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as plt
import help as hlp

# Loading data
#data = np.load('data100D.npy')
data = np.load('C:\\Users\\dyson\\Desktop\\UOT_assignments\\ECE1513\\a3\\data2D.npy')
[num_pts, dim] = np.shape(data)
k=3
x=tf.placeholder(tf.float32)
#index=tf.placeholder(tf.int32)
mu=tf.Variable(tf.random.truncated_normal([k,2],0.0,stddev=1.0,dtype=tf.float32))#mu
seta=tf.Variable(initial_value=1,dtype=tf.float32)
seta_square=tf.square(seta)#seta square
initial_prob=tf.random.truncated_normal([3],0.0,stddev=1.0,dtype=tf.float32)
prob=tf.Variable(initial_prob,dtype=tf.float32)
pi=hlp.logsoftmax(prob)#probability pi
soft_prob=[]
sum=0
for i in range(k):
    sum+=tf.exp(pi[i])
for i in range(k):
    soft_prob.append(tf.exp(pi[i])/sum)

d=2
storage=tf.Variable(initial_value=0.0)
for n in range(10000):
    cluster_prob=[]
    if n%10==0:
        print(n)
    for index in range(k):
        minus_item=x[n]-mu[index]#10000*3*2
        exp_item=tf.exp((-1/2)*minus_item*(1/seta_square)*minus_item)

        first_item=(1/tf.pow((2*math.pi),int(d/2))*tf.sqrt(seta_square) )

        pdf=first_item*exp_item

        possibility=soft_prob[index]*pdf
        cluster_prob.append(possibility)
    pos_sum=np.sum(cluster_prob)
    if n==0:
        storage=pos_sum
    else:
        storage=tf.multiply(storage,pos_sum)

loss=hlp.reduce_logsumexp(storage)*(-1)
optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.0001).minimize(loss)


init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for i in range(20):
        sess.run(optimizer,feed_dict={x:data})
        l=sess.run(loss,feed_dict={x:data})
        print(l)


