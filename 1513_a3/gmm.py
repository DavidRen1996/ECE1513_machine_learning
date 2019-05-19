import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as plt
import help as hlp

# Loading data
#data = np.load('data100D.npy')
data = np.load('C:\\Users\\dyson\\Desktop\\UOT_assignments\\ECE1513\\a3\\data2D.npy')
[num_pts, dim] = np.shape(data)

def cluster_pdf(x_input,mu_expand,seta):


    minus_item=x_input-mu_expand#10000*3*2
    d=2
    exp_item=tf.exp((-1/2)*minus_item*(1/seta)*minus_item)

    first_item=(1/tf.pow((2*math.pi),int(d/2))*tf.sqrt(seta) )

    pdf=first_item*exp_item
    return pdf
def cluster_prob(N_index,x_input,mu_expand,seta,k,k_desired):

    total_pdf=cluster_pdf(x_input,mu_expand,seta)
    sum=0
    for i in range(k):
        sum+=total_pdf[N_index,i,:]
    prob=total_pdf[N_index,k_desired,:]/sum
    return prob

seta=tf.constant(1,dtype=tf.float32)
Mu = tf.Variable(tf.random.truncated_normal([3, 2], 0.0, stddev=1.0, dtype=tf.float32))
mu_expand = tf.expand_dims(Mu, 0)
x=tf.placeholder(tf.float32)
x_expand=tf.expand_dims(x,dim=1)
totoal_pdf=cluster_prob(1,x_expand,mu_expand,seta,3,2)
log_pdf=hlp.reduce_logsumexp(totoal_pdf,reduction_indices=0)
optimizer=tf.train.AdamOptimizer(learning_rate=0.0001).minimize(log_pdf)
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    total = sess.run(log_pdf, feed_dict={x: data})
    print(total)
    pdf=sess.run(optimizer,feed_dict={x:data})
    #print(pdf.shape)
    #print(pdf[0,])





'''
def gaussian_distribution(x,u,seta):
    item_one=1/(tf.multiply((tf.sqrt(2*math.pi)),seta))
    minus_item=-(tf.square(x-u))/2*tf.square(seta)
    item_two=tf.exp(minus_item)
    return tf.multiply(item_one,item_two)
def log_possibility(x_input,K,u,seta):
    power_u=tf.pow(u,K)
    k_square=tf.square(K)
    power_seta=tf.pow(seta,k_square)
    return tf.log(gaussian_distribution(x_input,power_u,power_seta))
#def cluster_log_probability():
input=tf.placeholder(tf.float32)
log_ope=hlp.logsoftmax(input)
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    print(data[0,])
    k=sess.run(log_ope,feed_dict={input:data[0,]})
    print(k)
'''


'''
# For Validation set
if is_valid:
  valid_batch = int(num_pts / 3.0)
  np.random.seed(45689)
  rnd_idx = np.arange(num_pts)
  np.random.shuffle(rnd_idx)
  val_data = data[rnd_idx[:valid_batch]]
  data = data[rnd_idx[valid_batch:]]

# Distance function for GMM
def distanceFunc(X, MU):
    # Inputs
    # X: is an NxD matrix (N observations and D dimensions) 10000*2
    # MU: is an KxD matrix (K means and D dimensions) 3*2
    # Outputs
    # pair_dist: is the pairwise distance matrix (NxK) 10000*3
    # TODO

def log_GaussPDF(X, mu, sigma):
    # Inputs
    # X: N X D
    # mu: K X D
    # sigma: K X 1


    # log_pi: K X 1

    # Outputs:
    # log Gaussian PDF N X K

    # TODO

def log_posterior(log_PDF, log_pi):
    # Input
    # log_PDF: log Gaussian PDF N X K
    # log_pi: K X 1

    # Outputs
    # log_post: N X K

    # TODO
'''
