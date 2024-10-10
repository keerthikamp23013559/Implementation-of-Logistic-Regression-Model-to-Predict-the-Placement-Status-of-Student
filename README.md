# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. import required python library
2. load the data set and do the necessary preprocessing steps
3. use lable encoder to convert string into integer
4. split the data for training and testing
5. use logisticregression to do the classification 

## Program and Output:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: KEERTHIKA M P
RegisterNumber:  212223240071
*/
```
```
import numpy as np
import pandas as pd
data=pd.read_csv('Placement_data.csv')
data
d1=data.copy()
d1 
```
![image](https://github.com/user-attachments/assets/f7a7bc4c-a5cd-4b68-bf40-b55cc918975a)
```
data=data.drop(["sl_no","salary"],axis=1)
data
```
![image](https://github.com/user-attachments/assets/efd2210d-d195-4883-9b94-2cf2f8563710)
```
data.isnull().sum()

```
![image](https://github.com/user-attachments/assets/b3589ad4-2c80-459c-a4e7-575e835f146c)
```
data.duplicated().sum()
```
![image](https://github.com/user-attachments/assets/1ff251b8-cf46-4d65-9ed8-ab08228c9a01)

```
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["gender"]=le.fit_transform(data["gender"])
data["ssc_b"]=le.fit_transform(data["ssc_b"])
data["hsc_b"]=le.fit_transform(data["hsc_b"])
data["hsc_s"]=le.fit_transform(data["hsc_s"])
data["degree_t"]=le.fit_transform(data["degree_t"])
data["workex"]=le.fit_transform(data["workex"])
data["specialisation"]=le.fit_transform(data["specialisation"])
data["status"]=le.fit_transform(data["status"])
data
```
![image](https://github.com/user-attachments/assets/72e27dbe-963e-43d0-8540-b87c5fe8c387)
```
x=data.iloc[::-1]
x
```
![image](https://github.com/user-attachments/assets/c75cb4aa-dd7d-48fe-b205-8940041a3c86)
```
y=data["status"]
y
```
![image](https://github.com/user-attachments/assets/cbf265db-8dae-41a9-8b35-6afaf2b8bc91)
```
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver = "liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred
```
![image](https://github.com/user-attachments/assets/f682dad5-41ee-4df1-96a9-078212155e5a)

```
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
print("Accuracy:",accuracy)
```
![image](https://github.com/user-attachments/assets/e6710a57-cf59-4d95-97ce-5d678c2c678e)

```
from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test,y_pred)
print("Confusion Matrix:\n",confusion)
```
![image](https://github.com/user-attachments/assets/dbb68004-913b-4969-a4c0-cff49e13049b)

```
from sklearn.metrics import classification_report
classification=classification_report(y_test,y_pred)
print(classification)
```
![image](https://github.com/user-attachments/assets/d727ff22-b37f-4fae-a509-24c676625526)

```
lr.predict([[1,90,1,90,1,1,90,1,80,1,80,1,85]])
```
![image](https://github.com/user-attachments/assets/315b85f4-b504-4732-90ff-fc0889373e13)


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
