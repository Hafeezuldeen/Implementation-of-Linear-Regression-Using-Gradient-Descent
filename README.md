# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
1.Import the required library and read the dataframe.

2.Write a function computeCost to generate the cost function.

3.Perform iterations og gradient steps with learning rate.

4.Plot the Cost function using Gradient Descent and generate the required graph.
```
## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: HAFEEZUL DEEN S
RegisterNumber:  212223220028
*/
```
```
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
def linear_regression(X1,y,learning_rate = 0.1, num_iters = 1000):
    X = np.c_[np.ones(len(X1)),X1]
    theta = np.zeros(X.shape[1]).reshape(-1,1)
    
    for _ in range(num_iters):
        predictions = (X).dot(theta).reshape(-1,1)
        errors=(predictions - y ).reshape(-1,1)
        theta -= learning_rate*(1/len(X1))*X.T.dot(errors)
    return theta
data=pd.read_csv("50_Startups.csv")
data.head()
X=(data.iloc[1:,:-2].values)
X1=X.astype(float)
scaler=StandardScaler()
y=(data.iloc[1:,-1].values).reshape(-1,1)
X1_Scaled=scaler.fit_transform(X1)
Y1_Scaled=scaler.fit_transform(y)
print(X)
print(X1_Scaled)
theta= linear_regression(X1_Scaled,Y1_Scaled)
new_data=np.array([165349.2,136897.8,471784.1]).reshape(-1,1)
new_Scaled=scaler.fit_transform(new_data)
prediction=np.dot(np.append(1,new_Scaled),theta)
prediction=prediction.reshape(-1,1)
pre=scaler.inverse_transform(prediction)
print(prediction)
print(f"Predicted value: {pre}")
```

## Output:
Data Information


![311408466-e8b4b890-3dbe-46da-9b18-5b34ebaf02a0](https://github.com/Hafeezuldeen/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/144979314/e9b1fc7d-16f9-49f4-a3a6-1501751fc2a9)


Value of X


![311408478-d4da16e6-0ece-4ab1-8c5f-c4b197c7a512](https://github.com/Hafeezuldeen/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/144979314/8fe400bf-ef4d-4a65-bbf8-c4c193902a6c)




Predicted Value


![311408489-6364cf72-a2a4-4e81-9044-4aec92c8112a](https://github.com/Hafeezuldeen/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/144979314/7f5cba3d-289a-4fc6-9779-df0040429087)




## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
