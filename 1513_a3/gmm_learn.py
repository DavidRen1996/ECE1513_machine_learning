import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as plt
import help as hlp

# Loading data
#data = np.load('data100D.npy')
data = np.load('C:\\Users\\dyson\\Desktop\\UOT_assignments\\ECE1513\\a3\\data2D.npy')
[num_pts, dim] = np.shape(data)
seta=tf.constant([[0.5,0.3]],dtype=tf.float32)
log_pdf=hlp.reduce_logsumexp(seta)
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    total = sess.run(log_pdf)
    print(total)
    print(seta.eval())