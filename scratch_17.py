#Importing the libraries
import numpy as np
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('train.csv')

#missing data
dataset['Embarked']=dataset['Embarked'].fillna('S',inplace=False)
dataset['Age']=dataset['Age'].fillna(dataset['Age'].mean())

#Encoding (creating dummy variable)
dataset=pd.get_dummies(dataset,columns=['Embarked'],drop_first=True)
dataset=pd.get_dummies(dataset,columns=['Sex'],drop_first=True)
dataset=pd.get_dummies(dataset,columns=['Pclass'],drop_first=True)

X = dataset.iloc[:,[3,4,5,7,9,10,11,12,13]].values
y = dataset.iloc[:, 1].values

#Splitting train set into training and validation set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X[:,[0,3]] = sc_X.fit_transform(X[:,[0,3]])


# Fitting Simple kernel SVM classifier to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X, y)


####################PREPROCESSING THE TEST SET#######################################

# Importing the dataset
dataset2=pd.read_csv('test.csv')

#missing data
dataset2['Parch']=dataset2['Parch'].fillna(dataset2['Parch'].median())
dataset2['Fare']=dataset2['Fare'].fillna(dataset2['Fare'].median())
dataset2['Age']=dataset2['Age'].fillna(dataset2['Age'].mean())

#creating dummy variable(encoding)
dataset2=pd.get_dummies(dataset2,columns=['Embarked'],drop_first=True)
dataset2=pd.get_dummies(dataset2,columns=['Sex'],drop_first=True)
dataset2=pd.get_dummies(dataset2,columns=['Pclass'],drop_first=True)

X2 = dataset2.iloc[:,[2,3,4,6,8,9,10,11,12]].values

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X2[:,[0,3]] = sc_X.transform(X2[:,[0,3]])

# Predicting the Test set results
y_pred = classifier.predict(X2)

#to convert to csv files
y_pred = pd.Series(y_pred,name="Survived")

df = dataset2.copy()

df.columns.values
df = df.drop(['PassengerId','Name','Age','SibSp','Parch','Ticket','Fare','Cabin',
              'Embarked_Q','Embarked_S','Sex_male','Pclass_2','Pclass_3'],axis = 1)

df['PassengerId'] = np.arange(892,1310)
df['Survived'] = y_pred

df.to_csv('Survived.csv',index =False)

#to check how may incorrect or correct predictions you have by confusion matrix
'''from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)'''
