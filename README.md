# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Start the program

2.Import the python pandas library as pd

3.Read the contents of the Spam csv file

4.Display the first 5 rows of the dataset using head()

5.Assign x as v1 values and y as v2 values

6.From sklearn library select the feature extraction and import CountVectorizer

7.CountVectorizer will convert the Text to Numerical Data

8.From sklearn library import Support Vector Classifier (ie. SVC)

9.Predict the x_test using SVC

10.Print the accuracy of the SVM Model 11.Stop the program

## Program:
```
Program to implement the SVM For Spam Mail Detection..
Developed by: D.vishnu vardhan reddy
RegisterNumber:  212221230023
```
```
import chardet
file='/content/spam.csv'
with open(file, 'rb') as rawdata:
  result = chardet.detect(rawdata.read(100000))
result

import pandas as pd
data=pd.read_csv("/content/spam.csv",encoding = 'Windows-1252')

data.head()

data.info()

data.isnull().sum()

x=data["v1"].values

y=data["v2"].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()

x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)

from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
```

## Output:
1.Result output:

![image](https://github.com/JayanthYadav123/Implementation-of-SVM-For-Spam-Mail-Detection/assets/94836154/57703867-8c96-407d-b8d6-d96d30e2de25)

2.data.head():

![image](https://github.com/JayanthYadav123/Implementation-of-SVM-For-Spam-Mail-Detection/assets/94836154/527fada6-d94d-4db9-adef-b80b1d4820b2)

3.data.info():

![image](https://github.com/JayanthYadav123/Implementation-of-SVM-For-Spam-Mail-Detection/assets/94836154/c74ef3a7-a2b9-4187-b93b-c24711d62f29)

4.data.isnull().sum():

![image](https://github.com/JayanthYadav123/Implementation-of-SVM-For-Spam-Mail-Detection/assets/94836154/29e432e6-0c06-44f1-a128-da3692f41055)

5.Y_Prediction value:

![image](https://github.com/JayanthYadav123/Implementation-of-SVM-For-Spam-Mail-Detection/assets/94836154/82eec8fb-d803-4805-85a2-4e633798e7c2)

6.Accuracy value:

![image](https://github.com/JayanthYadav123/Implementation-of-SVM-For-Spam-Mail-Detection/assets/94836154/aebf5caa-9d0c-4fa7-af76-efee69880e3a)

## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
