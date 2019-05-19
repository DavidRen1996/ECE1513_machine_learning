import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as plt

data = np.load('C:\\Users\\dyson\\Desktop\\UOT_assignments\\ECE1513\\a3\\data100D.npy')
[num_pts, dim] = np.shape(data)
valid_batch = int(num_pts / 3.0)
np.random.seed(45689)
rnd_idx = np.arange(num_pts)
np.random.shuffle(rnd_idx)
val_data = data[rnd_idx[:valid_batch]]
data = data[rnd_idx[valid_batch:]]


def reduce_logsumexp(input_tensor, reduction_indices=1, keep_dims=False):
    """Computes the sum of elements across dimensions of a tensor in log domain.

       It uses a similar API to tf.reduce_sum.

    Args:
      input_tensor: The tensor to reduce. Should have numeric type.
      reduction_indices: The dimensions to reduce.
      keep_dims: If true, retains reduced dimensions with length 1.
    Returns:
      The reduced tensor.
    """
    max_input_tensor1 = tf.reduce_max(
        input_tensor, reduction_indices, keep_dims=keep_dims)
    max_input_tensor2 = max_input_tensor1
    if not keep_dims:
        max_input_tensor2 = tf.expand_dims(max_input_tensor2, reduction_indices)
    return tf.log(
        tf.reduce_sum(
            tf.exp(input_tensor - max_input_tensor2),
            reduction_indices,
            keep_dims=keep_dims)) + max_input_tensor1


def logsoftmax(input_tensor):
    """Computes normal softmax nonlinearity in log domain.

       It can be used to normalize log probability.
       The softmax is always computed along the second dimension of the input Tensor.

    Args:
      input_tensor: Unnormalized log probability.
    Returns:
      normalized log probability.
    """
    return input_tensor - reduce_logsumexp(input_tensor, reduction_indices=0, keep_dims=True)


def cluster_pdf(x_input, mu_expand, seta):
    minus_item = x_input - mu_expand  # 10000*3*2
    d = 2
    exp_item = tf.exp(-tf.square(minus_item) / (2 * seta))

    first_item = 1 / (2 * np.pi * tf.sqrt(seta))
    product = first_item * exp_item
    log_product = tf.log(product)
    pdf = tf.reduce_sum(log_product)
    return pdf


k = 15
epoch = 10
x = tf.placeholder(tf.float32)

mu = tf.Variable(tf.random.truncated_normal([k, 100], 0.0, stddev=1.0, dtype=tf.float32))  # mu

variance = tf.Variable(tf.random.truncated_normal([k], 0.0, stddev=1, dtype=tf.float32))
variance_square = tf.exp(variance)
pi_k = tf.Variable(tf.random.truncated_normal([k], 0.0, stddev=1, dtype=tf.float32))
prob_k = logsoftmax(pi_k)
prob_density = []
for i in range(k):
    density = cluster_pdf(x, mu[i], variance_square[i])
    prob_density.append(density)
prob_dense = tf.stack(prob_density)
cluster = tf.arg_max(prob_dense, 0)
pt = tf.stack([prob_dense, prob_k])
# px_mean=tf.zeros([1])
px_mean = tf.reduce_sum(pt, 0)
px = reduce_logsumexp(px_mean, reduction_indices=0, keep_dims=False)
L = -px
optimizer = tf.train.AdamOptimizer(learning_rate=0.00001).minimize(L)
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    loss_list = []
    assignment = []
    for i in range(epoch):

        total_loss = 0
        for num in range(6667):
            sess.run(optimizer, feed_dict={x: data[num]})
            # k = sess.run(prob_dense, feed_dict={x: data[num]})
            # print(k)
            l = sess.run(L, feed_dict={x: data[num]})

            total_loss += l
            if i == (epoch - 1):
                # dense = sess.run(prob_dense, feed_dict={x: data[num]})
                clusterk = sess.run(cluster, feed_dict={x: data[num]})
                assignment.append(clusterk)

        loss_list.append(total_loss)
        print(total_loss)
    for i in range(k):
        num=0
        for n in range((6667)):
            if assignment[n] == i:
                num += 1
        print(num)
    plt.figure(1)
    plt.plot(loss_list, color='RED')
    #plt.figure(2)
    #plt.scatter(data[:, 0], data[:, 1], c=assignment, s=50, alpha=0.5)

    plt.show()