import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
from sklearn.metrics import confusion_matrix


dataset = pd.read_csv('Churn_Modelling.csv')
x = dataset.iloc[:, 3:13].values   #independent variable matrix
y = dataset.iloc[:, 13].values  #class variable vector


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


json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
classifier_loaded = model_from_json(loaded_model_json)
classifier_loaded.load_weights("model.h5")


new_prediction = classifier_loaded.predict(sc.transform(np.array([[0.0,0,600,1,40,3,60000,2,1,1,50000]])))


new_prediction = (new_prediction > 0.5)

print(new_prediction)

