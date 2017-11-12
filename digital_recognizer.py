#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 12:08:39 2017

@author: apple
"""
"""Digital Recognizer"""
from time import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

"""Data preparation part
    reshape pixel to matrix of image 
"""
start=time()
def reshape_to_1d(array):
    return  array.reshape((array.size,1))

train=pd.read_csv("train.csv",delimiter=",")
test=pd.read_csv("test.csv",delimiter=",")

#train_label=np.array(train["label"]) #training label
train_pixel_index=[("pixel"+str(i)) for i in range(train.shape[1]-1)]
train_pixel=train[train_pixel_index]
#Store image in train_img
train_img=[0]*train_pixel.shape[0] #training data

for i in range(len(train_img)):
    #Reshape
    img_pixel=np.array((train_pixel.loc[i,:]))
    #img_pixel=img_pixel.reshape((28,28))
    train_img[i]=img_pixel
    
    
neigh=KNeighborsClassifier(n_neighbors=10)
neigh.fit(train_img,train["label"])
predictions=neigh.predict(test)
#print("EVALUATION ON TESTING DATA")
#print(classification_report(train, predictions))

end=time()
print("Process time :",end-start,"sec") #Let see our process time uses!

#Write to csv file name "submission"
submission=pd.DataFrame()
submission["ImageId"]=np.arange(1,test.shape[0]+1)
submission["Label"]=predictions
submission.to_csv("submission.csv",encoding="utf-8",index=False)

