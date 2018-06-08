import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


dataset = pd.read_csv('Churn_Modelling.csv')
x = dataset.iloc[:, 3:13].values   #independent variable matrix
y = dataset.iloc[:, 13].values  #class variable vector

#print(x)



from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
x[:, 1] = labelencoder_X_1.fit_transform(x[:, 1])   #encoding country column to numbers

labelencoder_X_2 = LabelEncoder()
x[:, 2] = labelencoder_X_2.fit_transform(x[:, 2])   #encoding male female column



onehotencoder = OneHotEncoder(categorical_features = [1])
x = onehotencoder.fit_transform(x).toarray()    #dummy variable for country
x=x[:,1:]   #removing one column to avaoid dummy variable trap



from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)    #splitting data into training and test set


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train) #feature scaling
x_test = sc.transform(x_test)   #feature scaling

import keras
from keras.models import Sequential
from keras.layers import Dense

#initializing the NN and adding layers
classifier = Sequential()

#first hidden layer
classifier.add(Dense(units=6,kernel_initializer='uniform',activation='relu',input_dim=11))

#second hidden layer
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))

#output layer using sigmoid activation since it can help us return probabaility
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

#using adam optimizer which is a type of socastic descent. and using binary_crossentropy
#which is for binary outputs such as this one , if we had categoical variable we would use categorical_crossentropy
classifier.compile(optimizer='adam',loss='binary_crossentropy', metrics=['accuracy'])

#fitting dataset into ANN training and test set
classifier.fit(x_train,y_train, batch_size=10, nb_epoch=100)

y_pred = classifier.predict(x_test)
y_pred = (y_pred > 0.5)

print(y_pred)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

print(cm)

