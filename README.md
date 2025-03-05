# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. import the needed packages.
2. Assigning hours to x and scores to y.
3. Plot the scatter plot.
4. Use mse,rmse,mae formula to find the values.

## Program:
```
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: NAVEEN G
RegisterNumber:  212223220066


import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error,mean_squared_error
import matplotlib.pyplot as plt

dataset=pd.read_csv('student_scores.csv')
print(dataset.head())
dataset=pd.read_csv('student_scores.csv')
print(dataset.tail())
x=dataset.iloc[:,:-1].values
print(x)
y=dataset.iloc[:,1].values
print(y)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(x_train,y_train)
y_pred = reg.predict(x_test)
print(y_pred)
print(y_test)

plt.scatter(x_train,y_train,color='purple')
plt.plot(x_train,reg.predict(x_train),color='black')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

plt.scatter(x_test,y_test,color='red')
plt.plot(x_train,reg.predict(x_train),color='black')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

mse=mean_absolute_error(y_test,y_pred)
print('Mean Square Error = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('Mean Absolute Error = ',mae)
rmse=np.sqrt(mse)
print("Root Mean Square Error = ",rmse)


```

## Output:
Head Values

<img width="1090" alt="319854949-4ddf5d62-c261-42be-8b67-6f6df35f3d36" src="https://github.com/user-attachments/assets/5b3d83cc-bdee-4771-aaa2-a53174ee95a6" />

Tail values

<img width="1090" alt="319855044-dfa9fd82-e723-4ed4-aaec-ad66ffa348e1" src="https://github.com/user-attachments/assets/5eeb8622-e3d5-4967-8613-f12d14588f19" />

Compare Dataset


<img width="1090" alt="319855079-9118849a-2323-4b9c-a362-3dc8d060587a" src="https://github.com/user-attachments/assets/fa3761bb-4c96-4f38-9f0d-70f1373ce526" />

Predication values of X and Y

<img width="1090" alt="319855219-5ed7921d-08e1-408e-8383-6899933b01ee" src="https://github.com/user-attachments/assets/1c8f1928-3258-41ca-8a30-21097d760c04" />

Training set


<img width="1090" alt="319855290-9fae6c7e-e00d-449f-8f16-cbbe94095e76" src="https://github.com/user-attachments/assets/3bd0d892-2b04-4b6e-9c6b-639bf21bba42" />

Testing Set

<img width="1090" alt="319855335-bea6f3d4-6bf1-4c28-b7ff-167a07ec1d30" src="https://github.com/user-attachments/assets/953489af-1e96-4c66-86de-798e6cb0de38" />

MSE,MAE and RMSE

<img width="1090" alt="319855368-0f1f6538-a2a8-4c7c-838e-8ccbca0c7b6c" src="https://github.com/user-attachments/assets/294fb7f6-87ba-4300-bb87-28601008cde9" />



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
