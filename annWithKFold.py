import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense


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


from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu', input_dim=11))
    classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu'))
    classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))
    classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return classifier

classifier = KerasClassifier(build_fn=build_classifier, batch_size = 10, epochs = 100)

accuracies = cross_val_score(estimator=classifier, X = x_train, y=y_train, cv=10, n_jobs=-1)

print(accuracies)

mean = accuracies.mean()

variance = accuracies.std()

print(mean)
print(accuracies)