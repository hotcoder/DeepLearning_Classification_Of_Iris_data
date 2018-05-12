#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 12 17:36:39 2018

@author: z002krv
"""

import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV
from keras.models import Sequential
from keras.layers.core import Dense,Activation
from keras.utils import np_utils

#Load iris data from seaborn
def load_iris_data():
    iris = sns.load_dataset('iris')
    return iris

iris = load_iris_data();

#print the the first five rows of the iris data

print(iris.head(5))

#Visualize the data
print(sns.pairplot(iris,hue='species'))