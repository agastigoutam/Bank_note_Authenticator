# Bank Note Authentication
#Data were extracted from images that were taken from genuine and forged banknote-like specimens. 
# For digitization, an industrial camera usually used for print inspection was used. The final images have 400x 400 pixels.
# Due to the object lens and distance to the investigated object gray-scale pictures with a resolution of about 660 dpi were gained. 
# Wavelet Transform tool were used to extract features from images

import pandas as pd
import numpy as np
import sklearn
import pickle



df=pd.read_csv('C:/Users/ASUS/Desktop/All/GIT_Projects/Bank_Note_Authentication_App/BankNote_Authentication.csv')
# print(df.head(5))
# taking dependent and independent variables 
X=df.iloc[:,:-1]
y=df.iloc[:,-1]

### Spliting dataset for training and testing purposes
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

### Implementing Random Forest classifier and traing model on training dataset
from sklearn.ensemble import RandomForestClassifier
classifier=RandomForestClassifier()
classifier.fit(X_train,y_train)

# Checking for prediction on testing datapart
y_predict=classifier.predict(X_test)

# checking the accuracy 
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_predict)
# print(accuracy)

#  create a pickle file 
pickle_out=open("C:/Users/ASUS/Desktop/All/GIT_Projects/Bank_Note_Authentication_App/classifier.pkl","wb")
pickle.dump(classifier,pickle_out)
pickle_out.close()

















