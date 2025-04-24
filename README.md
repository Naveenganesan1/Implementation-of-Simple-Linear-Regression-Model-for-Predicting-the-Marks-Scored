# Ex-2:Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.
## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
## import the needed packages.
## Assigning hours to x and scores to y.
## Plot the scatter plot.
## Use mse,rmse,mae formula to find the values.

## Program:

```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Naveen G
Register Number: 212223220066
*/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
print(df)

df.head(0)
df.tail(0)
print(df.head())
print(df.tail())

x = df.iloc[:,:-1].values
print(x)

y = df.iloc[:,1].values
print(y)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
print(y_pred)
print(y_test)

#Graph plot for training data
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

#Graph plot for test data
plt.scatter(x_test,y_test,color='black')
plt.plot(x_train,regressor.predict(x_train),color='red')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)

mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)

rmse=np.sqrt(mse)
print("RMSE= ",rmse)
```
## Output:
### Dataset

![dataset](https://github.com/user-attachments/assets/d35c2895-396c-454d-8ee8-4eac1e646cc2)





### Head Values


![head](https://github.com/user-attachments/assets/61a3531d-4cc6-48a5-a4fc-2245fdf1244e)




### Tail Values



![tail](https://github.com/user-attachments/assets/ff3cbf51-e5e5-4d19-9c0d-a121b2e825ba)


### X and Y values


![xyvalues](https://github.com/user-attachments/assets/25300cba-aa89-4183-9b5d-a9711709930b)




### Predication values of X and Y

![predict ](https://github.com/user-attachments/assets/fd9cf9a6-4f11-465e-8c94-67c38aa0e065)



### MSE,MAE and RMSE

![values](https://github.com/user-attachments/assets/17e8c95f-8a69-47a3-8da2-ebb9e480809a)



### Training Set

![train](https://github.com/user-attachments/assets/49fab058-b0d3-4056-be11-0c969e7619ca)


### Testing Set

![test](https://github.com/user-attachments/assets/f91958bd-ca67-4b82-addb-7d3fb6c96290)



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
