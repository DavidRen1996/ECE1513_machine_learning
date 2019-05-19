from helper import helpers
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
import os

class back_propagation_derivative(helpers):
    def __init__(self):
        print('initialize class back propagation')

    def grad_out_weight(self,target,predictions,xhid):

        grad_ow=np.dot((predictions.T-target).T,xhid.T)
        return grad_ow.T
    def grad_out_bias(self,target,predictions):
        one_matrix=np.ones([target.shape[0],1])

        grad_ob=np.dot((predictions.T-target).T,one_matrix)
        #return 1*10
        return grad_ob.T
    def grad_hid_weight(self,weight_out,target,predictions,x_zero,s_hid):
        grad=np.dot((predictions.T-target),weight_out.T)
        #s_hid=helpers.compute(self,weight_hid,x_zero,bias_hid)
        grad_RELU=self.grad_relu(s_hid)
        grad1=grad*grad_RELU.T
        grad_hw=np.dot(grad1.T,x_zero)
        return grad_hw.T
    def grad_hid_bias(self,target,predictions,weight_out,s_hid):
        seta=(predictions.T-target)
        grad=np.dot(seta,weight_out.T)
        #s_hid=helpers.compute(self,weight_hid,x_zero,bias_hid)
        grad_RELU = self.grad_relu(s_hid)
        ones=np.ones([1,10000])
        grad_hb=np.dot((grad*grad_RELU.T).T,ones.T)

        return grad_hb.T

    def grad_relu(self, x):

        return np.where(x > 0, 1.0, 0.0)
    '''def grad_relu(self,x):
        minus = np.where(x < 0)
        x[minus] = 0
        pos=np.where(x > 0)
        x[pos]=1
        return x'''












#call the helper file will execute the init function in helpers class for once
'''helps=helpers()
trainD,validD,testD,trainT,validT,testT=helps.loadData()
new_train,new_valid,new_test=helps.convertOneHot(trainT,validT,testT)'''
