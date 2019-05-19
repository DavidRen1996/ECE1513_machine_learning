from helper import helpers
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
import os
from back_propagation import back_propagation_derivative

class learning(back_propagation_derivative,helpers):
    def __init__(self):
        print('learning initiated')

    def forward_propagation(self,x_zero,weight_hid,bias_hid,weight_out,bias_out):
        s_hid=helpers.compute(self,weight_hid.T,x_zero.T,bias_hid.T)

        x_hid=helpers.relu(self,s_hid)

        s_out=helpers.compute(self,weight_out.T,x_hid,bias_out.T)
        x_out=helpers.softmax(self,s_out)

        return x_out,s_out,x_hid,s_hid

    def accuracy_compute(self,target,prediction):
        rows=target.shape[0]
        sum=0
        for i in range(rows):
            labels=np.argmax(target[i])
            pre_label=np.argmax(prediction[i])
            if labels==pre_label:
                sum=sum+1

        return sum/rows

    def back_propagation(self,target,weight_out,x_zero,weight_hid,bias_hid,bias_out):
        predict,so,xh,sh=self.forward_propagation(x_zero,weight_hid,bias_hid,weight_out,bias_out)


        grad_ow=back_propagation_derivative.grad_out_weight(self,target,predict,xh)
        grad_ob=back_propagation_derivative.grad_out_bias(self,target,predict)
        grad_hw=back_propagation_derivative.grad_hid_weight(self,weight_out,target,predict,x_zero,sh)
        grad_hb = back_propagation_derivative.grad_hid_bias(self, target, predict, weight_out, sh)
        #return 10000*10
        #print(sh)
        return grad_ow,grad_ob,grad_hw,grad_hb,predict,so,xh,sh
    def run(self,validD,testD,new_valid,new_test,target,weight_out,x_zero,weight_hid,bias_hid,bias_out):
        w_HIDLAYER=weight_hid
        w_OUTLAYER=weight_out
        b_HIDLAYER=bias_hid
        b_OUTLAYER=bias_out
        v_hid = np.ones([784, 1000]) / 10**5
        v_out = np.ones([1000, 10]) / 10**5
        index=[]
        accurate=[]
        loss_matrix=[]
        loss_matrix_valid=[]
        loss_matrix_test=[]
        accuracy_valid_list=[]
        accuracy_test_list = []
        for i in range(200):

            gradient_ow,gradient_ob,gradient_hw,gradient_hb,prediction,s_out,x_hid,s_hid=self.back_propagation(target,w_OUTLAYER,x_zero,w_HIDLAYER,b_HIDLAYER,b_OUTLAYER)
            x_valid_out, s_valid_out, x_valid_hid, s_valid_hid=self.forward_propagation(validD,w_HIDLAYER,b_HIDLAYER,w_OUTLAYER,b_OUTLAYER)
            x_test_out, s_test_out, x_test_hid, s_test_hid = self.forward_propagation(testD, w_HIDLAYER,b_HIDLAYER, w_OUTLAYER,b_OUTLAYER)
            if i == 15:
                print('help')
            v_hid=0.9*v_hid+0.00002*gradient_hw
            v_out=0.9*v_out+0.00002*gradient_ow

            w_HIDLAYER=w_HIDLAYER-v_hid
            w_OUTLAYER=w_OUTLAYER-v_out
            b_HIDLAYER = b_HIDLAYER-0.00001*gradient_hb
            b_OUTLAYER = b_OUTLAYER-0.00001*gradient_ob


            #print(loss)

            #if i%5==0 and i!=0:
            print ('sample',i)
            loss = helpers.averageCE(self, target, prediction.T)
            loss_test= helpers.averageCE(self,new_test,x_test_out.T)
            loss_valid = helpers.averageCE(self, new_valid, x_valid_out.T)
                #loss=helpers.averageCE(self,target,prediction.T)
            #print('train error:',loss)
            loss_matrix.append(loss)
            loss_matrix_valid.append(loss_valid)
            print('valid error:', loss_valid)
            loss_matrix_test.append(loss_test)
            print('test error:', loss_test)
            index.append(i)
            accuracy=self.accuracy_compute(target,prediction.T)
            accuracy_test = self.accuracy_compute(new_test, x_test_out.T)
            accuracy_valid = self.accuracy_compute(new_valid, x_valid_out.T)
            accurate.append(accuracy)
            accuracy_valid_list.append(accuracy_valid)
            accuracy_test_list.append(accuracy_test)
            print('train accuracy:',accuracy)
            print('valid accuracy:',accuracy_valid)
            print('test accuracy:',accuracy_test)
        print('plot figure')
        print(loss_matrix,index)

        plt.figure(1)

        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title('Neural Network Loss')
        plt.plot(loss_matrix, color='RED',label="Train Data")
        plt.plot(loss_matrix_valid, color='Green',label="Valid Data")
        plt.plot(loss_matrix_test, color='Black',label="Test Data")
        plt.legend(loc=2, bbox_to_anchor=(1.05, 1.0), borderaxespad=0.)
        plt.figure(2)
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title('Neural Network Accurary')
        plt.plot(accurate, color='RED',label="Train Data")
        plt.plot(accuracy_valid_list, color='Green',label="Valid Data")
        plt.plot(accuracy_test_list, color='Black',label="Test Data")
        plt.legend(loc=2, bbox_to_anchor=(1.05, 1.0), borderaxespad=0.)
        plt.show()
        return loss_matrix
















weight_hid_initial=np.random.normal(0,2/1784,[784,1000])
weight_out_initial=np.random.normal(0,2/1010,[1000,10])
bias_hid=np.random.normal(0,2/1000,[1,1000])
bias_out=np.random.normal(0,2/10,[1,10])



helps=helpers()
trainD,validD,testD,trainT,validT,testT=helps.loadData()
new_train,new_valid,new_test=helps.convertOneHot(trainT,validT,testT)
learn=learning()
reshaped_trainD=helps.reshape_datamatrix(trainD)
reshaped_validD=helps.reshape_datamatrix(validD)
reshaped_testD=helps.reshape_datamatrix(testD)
#forward_result=learn.forward_propagation(reshaped_trainD,weight_hid_initial,bias_hid,weight_out_initial,bias_out)
'''grad_ow,grad_ob,grad_hw,grad_hb=learn.back_propagation(new_train,weight_out_initial,reshaped_trainD,weight_hid_initial,bias_hid,bias_out)
print(grad_ow.shape)
print(grad_ob.shape)
print(grad_hw.shape)
print(grad_hb.shape)'''


loss=learn.run(reshaped_validD,reshaped_testD,new_valid,new_test,new_train,weight_out_initial,reshaped_trainD,weight_hid_initial,bias_hid,bias_out)