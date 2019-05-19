import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
import os
class helpers:
    def __init__(self):
        print('initialized')

    def loadData(self):
        with np.load("notMNIST.npz") as data:
            Data, Target = data["images"], data["labels"]
            np.random.seed(521)
            randIndx = np.arange(len(Data))
            np.random.shuffle(randIndx)
            Data = Data[randIndx] / 255.0
            Target = Target[randIndx]
            trainData, trainTarget = Data[:10000], Target[:10000]
            validData, validTarget = Data[10000:16000], Target[10000:16000]
            testData, testTarget = Data[16000:], Target[16000:]
        return trainData, validData, testData, trainTarget, validTarget, testTarget
    #data structure description, data is 3 dimensions matrix (10000,28,28)
    #target is number ranges from 0 to 9, 2 dimensions (10000,1)

    # Implementation of a neural network using only Numpy - trained using gradient descent with momentum
    def convertOneHot(self,trainTarget, validTarget, testTarget):
        newtrain = np.zeros((trainTarget.shape[0], 10))
        newvalid = np.zeros((validTarget.shape[0], 10))
        newtest = np.zeros((testTarget.shape[0], 10))

        for item in range(0, trainTarget.shape[0]):
            newtrain[item][trainTarget[item]] = 1
        for item in range(0, validTarget.shape[0]):
            newvalid[item][validTarget[item]] = 1
        for item in range(0, testTarget.shape[0]):
            newtest[item][testTarget[item]] = 1
        return newtrain, newvalid, newtest
    #the converted matrixes are (10000,10), in 10 column eacj column is 0 unless it is the target label
    #e.g:target=7 will have [0,0,0,0,0,0,0,1,0,0]

    def shuffle(self,trainData, trainTarget):
        np.random.seed(421)
        randIndx = np.arange(len(trainData))
        #create an array range from 0 to 10000
        target = trainTarget
        np.random.shuffle(randIndx)
        #shuffle the array
        data, target = trainData[randIndx], target[randIndx]
        #shuffle the trainning data and target with the shuffled array
        return data, target

    def relu(self, x):

        return np.maximum(0,x)
    '''def relu(self,x):
        minus=np.where(x<0)

        #find all positions where x equals minus and positive
        #replace value with relu
        x[minus]=0
        print('finish')
        return x'''
    def reshape_datamatrix(self,data_matrix):
        lines=data_matrix.shape[0]
        new_data=np.reshape(data_matrix,[lines,-1])
        return new_data

    def softmax(self,data_matrix):

        soft_func=np.exp(data_matrix)/np.sum(np.exp(data_matrix),axis=0,keepdims=True)
        #if want to add the columns use axis=1,keepdims=True
        return soft_func

    def compute(self,weight,data,bias):
        result=np.dot(weight,data)+bias
        return result

    def averageCE(self,target, prediction):
        # print("prediction")
        # print(np.log(prediction[0]))
        L_sum = np.sum(target * np.log(prediction + 10 ** -10))
        n = target.shape[0]
        L = (-1) * (1. / n) * L_sum
        return L
    '''def averageCE(self,target,predict):
        soft_result=self.softmax(predict)
        log_compute=np.log(soft_result)
        size=target.shape[0]
        #* multiply the corresponding places, while np.dot is standard matrix multiply
        production=(target)*log_compute
        loss=(-1/size)*np.sum(production)
        return loss'''
    def gradCE(self,target,out_prediction):
        gradient=np.mean(out_prediction-target)
        return gradient
        '''soft_result=self.softmax(out_prediction)
        grad=(-target/soft_result)+(target*soft_result)
        return grad'''








'''helps=helpers()
trainD,validD,testD,trainT,validT,testT=helps.loadData()
new_train,new_valid,new_test=helps.convertOneHot(trainT,validT,testT)
#sfunction=helps.softmax(new_train)
#error=helps.averageCE(new_train,new_train)
gra=helps.gradCE(new_train,new_train)
print(gra.shape,gra[0])

#helps.reshape_datamatrix(trainD)
#da,tar=helps.shuffle(trainD,trainT)'''





'''
#relu test
x=np.array([-1,2,-9,8])
k=helps.relu(x)
print(k)
'''



'''
#data structure test
print(trainD.shape)
print(testT[0])
train_new,valid_new,test_new=helps.convertOneHot(trainT,validT,testT)
print (test_new.shape)
print(test_new[0],test_new[1],test_new[2],test_new[3])'''