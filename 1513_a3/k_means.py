import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import help as hlp

# Loading data
data = np.load('C:\\Users\\dyson\\Desktop\\UOT_assignments\\ECE1513\\a3\\data2D.npy')
#data = np.load('data100D.npy')
[num_pts, dim] = np.shape(data)


def k_means(K,data):
    np.random.seed(421)
    #u=tf.Variable(tf.random.truncated_normal([K,2],0.0,stddev=1.0,dtype=tf.float32))
    u = tf.Variable(tf.random_normal([K, 2], 0.0, stddev=1.0, dtype=tf.float32))
    u_expand=tf.expand_dims(u,0)#expand dimension to substract
    x=tf.placeholder(tf.float32,shape=[10000,2],name='x')
    x_expand=tf.expand_dims(x,dim=1)
    minus=x_expand-u_expand
    loss=tf.reduce_sum(tf.square(minus),axis=2)
    bump=tf.arg_min(loss,1)#get the bump number that produce the minimal loss

    bump_loss=tf.reduce_min(loss,axis=1)#get the minimal bump loss
    total_loss=tf.reduce_mean(bump_loss)#total average loss
    optimizer=tf.train.AdamOptimizer(learning_rate=0.01, beta1=0.9, beta2=0.99, epsilon=1e-5).minimize(total_loss)
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        loss_list=[]
        sess.run(init)
        center = sess.run(u_expand, feed_dict={x: data})
        print(center)
        for index in range(900):

            opt=sess.run(optimizer,feed_dict={x: data})
            loss=sess.run(total_loss,feed_dict={x: data})
            loss_list.append(loss)
            if index==899:
                bump=  sess.run(bump,feed_dict={x: data})
                final_center=sess.run(u_expand,feed_dict={x: data})
                reshaped_final_center=np.reshape(final_center,[K,2])
                print(reshaped_final_center.shape)
        plt.figure(1)
        plt.plot(loss_list, color='RED')
        plt.figure(2)
        plt.scatter(data[:, 0], data[:, 1], c=bump, s=50, alpha=0.5)

        plt.plot(reshaped_final_center[:, 0], reshaped_final_center[:, 1], 'kx', markersize=15)

        plt.show()
        '''
        u0=sess.run(u_expand)
        n=sess.run(bump_loss,feed_dict={x:data})
        l=sess.run(loss,feed_dict={x:data})
        #bl = sess.run(bump_loss, feed_dict={x: data})
        print(n[0,])
        print(l[0,])
        #print(bl[0,])

        #print(u.eval())

        '''


k_means(5,data)





'''
# For Validation set
if is_valid:
  valid_batch = int(num_pts / 3.0)
  np.random.seed(45689)
  rnd_idx = np.arange(num_pts)
  np.random.shuffle(rnd_idx)
  val_data = data[rnd_idx[:valid_batch]]
  data = data[rnd_idx[valid_batch:]]


# Distance function for K-means
def distanceFunc(X, MU):
    # Inputs
    # X: is an NxD matrix (N observations and D dimensions)
    # MU: is an KxD matrix (K means and D dimensions)
    # Outputs
    # pair_dist: is the pairwise distance matrix (NxK)
    # TODO

'''
