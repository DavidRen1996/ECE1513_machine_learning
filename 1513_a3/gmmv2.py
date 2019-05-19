import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as plt
import help as hlp
data = np.load('C:\\Users\\dyson\\Desktop\\UOT_assignments\\ECE1513\\a3\\data2D.npy')
[num_pts, dim] = np.shape(data)
def cluster_pdf(x_input,mu_expand,seta):

    minus_item=x_input-mu_expand#10000*3*2
    d=2
    exp_item=tf.exp((-1/2)*tf.square(minus_item)/seta)

    first_item=1/(2*np.pi*tf.sqrt(seta))

    pdf=tf.reduce_sum(first_item*exp_item)
    return pdf
k=3
epoch=2
x=tf.placeholder(tf.float32)

mu=tf.Variable(tf.random.truncated_normal([k,2],0.0,stddev=1.0,dtype=tf.float32))#mu

variance=tf.Variable(tf.random.truncated_normal([k],0.0,stddev=1,dtype=tf.float32))
variance_square=tf.exp(variance)
pi_k=tf.Variable(tf.random.truncated_normal([k],0.0,stddev=1,dtype=tf.float32))
prob_k=hlp.logsoftmax(pi_k)
prob_density=[]
for i in range(k):
    density=cluster_pdf(x,mu[i],variance_square[i])
    prob_density.append(tf.log(density))
prob_dense=tf.stack(prob_density)
cluster=tf.arg_max(prob_dense,0)
pt=tf.stack([prob_dense,prob_k])
#px_mean=tf.zeros([1])
px_mean=tf.reduce_sum(pt,0)
px=hlp.reduce_logsumexp(px_mean,reduction_indices=0,keep_dims=False)
L=-px
optimizer=tf.train.AdamOptimizer(learning_rate=0.00001).minimize(L)
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    loss_list=[]
    assignment=[]
    for i in range(epoch):

        total_loss=0
        for num in range(10000):
            sess.run(optimizer, feed_dict={x: data[num]})
            #l = sess.run(pi_k, feed_dict={x: data[num]})

            l=sess.run(L,feed_dict={x:data[num]})

            total_loss+=l
            if i==(epoch-1):
                #dense = sess.run(prob_dense, feed_dict={x: data[num]})
                c = sess.run(cluster, feed_dict={x: data[num]})
                assignment.append(c)
            if num%1000==0:
                print('iteration:',num,'loss:',total_loss)

        loss_list.append(total_loss)
        print(total_loss)

    plt.figure(1)
    plt.plot(loss_list, color='RED')
    plt.figure(2)
    plt.scatter(data[:, 0], data[:, 1], c=assignment, s=50, alpha=0.5)



    plt.show()