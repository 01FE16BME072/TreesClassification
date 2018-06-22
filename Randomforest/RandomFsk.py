import pandas as pd
import numpy as np
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier

dataframe = pd.read_csv('cancer.csv')
dataframe.replace('?',-99999,inplace = True)
dataframe.drop(['id'],1,inplace = True)

X = dataframe.drop(['CLass'],1)
Y = dataframe['CLass']

X_train,X_test,Y_train,Y_test = cross_validation.train_test_split(X,Y,test_size = 0.25)

classifier = RandomForestClassifier(n_estimators = 10,criterion = 'entropy',random_state = 0)
classifier.fit(X_train,Y_train)

print 'Accuracy:',classifier.score(X_test,Y_test)

PredictThis = np.array([4,1,1,2,3,1,6,7,3])
PredictThis = PredictThis.reshape(1,-1)

Predict = classifier.predict(PredictThis)
print(Predict)