# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 21:10:57 2018

@author: This pc

"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing the dataset
dataset = pd.read_csv("Churn_Modelling.csv")
x = dataset.iloc[:,3:13].values
y = dataset.iloc[:,13].values

#encode categorical values
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_x1 = LabelEncoder()
onehotencoder = OneHotEncoder(categorical_features = [1])
x[:,1] = labelencoder_x1.fit_transform(x[:,1])
labelencoder_x2 = LabelEncoder()
x[:,2] = labelencoder_x2.fit_transform(x[:,2])
x = onehotencoder.fit_transform(x).toarray()

#delete one row to avoid dummy variable trap
x = x[:,1:]

#split into test and train
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y, test_size = 0.2, random_state = 0)

#feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

#we implement kfold
#kfold belongs to sklearn, so need to wrap it to keras
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense

def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(units=6, kernel_initializer="uniform", activation="relu", input_dim=11))
    classifier.add(Dense(units=6, kernel_initializer="uniform", activation="relu"))
    classifier.add(Dense(units=1, kernel_initializer="uniform", activation="sigmoid", input_dim=11))
    classifier.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy', metrics = ['accuracy'] )
    return classifier
#we need to create an object of kerasclassifier class
#classifier is local inside build_classifier, we can initialize a new global
classifier = KerasClassifier(build_fn = build_classifier, batch_size = 25, epochs = 500)
accuracies = cross_val_score(estimator = classifier, X = x_train, y = y_train, cv = 10)

#mean of accuracy
mean = accuracies.mean()
variance = accuracies.std()


#Tuning the ANN
#kernal restarted - the part above not run- the following is the second part
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense

def build_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(units=6, kernel_initializer="uniform", activation="relu", input_dim=11))
    classifier.add(Dense(units=6, kernel_initializer="uniform", activation="relu"))
    classifier.add(Dense(units=1, kernel_initializer="uniform", activation="sigmoid", input_dim=11))
    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'] )
    return classifier
#we need to create an object of kerasclassifier class
#classifier is local inside build_classifier, we can initialize a new global
classifier = KerasClassifier(build_fn = build_classifier)
#dictionary of hyper parameters
parameters = {'batch_size' : [25,32],
              'epochs' : [100,500],
              'optimizer' : ['adam', 'rmsprop']}

grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10)
grid_search = grid_search.fit(x_train, y_train)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_













