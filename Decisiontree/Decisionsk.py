import pandas as pd
import numpy as np
from sklearn import cross_validation
from sklearn.tree import DecisionTreeClassifier

dataframe = pd.read_csv('cancer.csv')
#print(dataframe.head())
dataframe.replace('?',-99999,inplace = True)
dataframe.drop(['id'],1,inplace = True)

X = dataframe.drop(['CLass'],1)
Y = dataframe['CLass']

X_train,X_test,Y_train,Y_test = cross_validation.train_test_split(X,Y,test_size = 0.25)

classifier = DecisionTreeClassifier(criterion = 'entropy',random_state = 0)
classifier.fit(X_train,Y_train)

Accuracy = classifier.score(X_test,Y_test)
print 'Accuracy:',Accuracy

PredictThis = np.array([4,1,1,2,3,5,6,7,4])
PredictThis = PredictThis.reshape(1,-1)

Predict = classifier.predict(PredictThis)

print(Predict)