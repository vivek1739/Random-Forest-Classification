# -*- coding: utf-8 -*-

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

# getting path of raw data
rawdata_path = os.path.join(os.path.pardir,'data','raw')
processeddata_path = os.path.join(os.path.pardir,'data','processed')
dataset = pd.read_csv(os.path.join(rawdata_path,'Social_Network_Ads.csv'))

# getting X and y
X = dataset.iloc[:,1:4].values
y = dataset.iloc[:,4].values

# encoding the gender and removing dummy variable
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelEncoder = LabelEncoder()
X[:,0] = labelEncoder.fit_transform(X[:,0])

onehotencoder = OneHotEncoder(categorical_features=[0])
X = onehotencoder.fit_transform(X).toarray()
X = X[:,1:]

# Splitting the dataset into the training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

# Apply feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Save the processed data in data/processed
np.savetxt(os.path.join(processeddata_path,'X_train.csv'), X_train, delimiter=",")
np.savetxt(os.path.join(processeddata_path,'X_test.csv'), X_test, delimiter=",")
np.savetxt(os.path.join(processeddata_path,'y_train.csv'), y_train, delimiter=",")
np.savetxt(os.path.join(processeddata_path,'y_test.csv'), y_test, delimiter=",")