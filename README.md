# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

Neural networks consist of simple input/output units called neurons. In this article, we will see how neural networks can be applied to regression problems.

Regression helps in establishing a relationship between a dependent variable and one or more independent variables. Although neural networks are complex and computationally expensive, they are flexible and can dynamically pick the best type of regression, and if that is not enough, hidden layers can be added to improve prediction.

Build your training and test set from the dataset, here we are making the neural network 3 hidden layer with activation layer as relu and with their nodes in them. Now we will fit our dataset and then predict the value.

## Neural Network Model


## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM
### Name: Meenakshi M
### Reg.No: 212221230057
```py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from google.colab import auth
import gspread
from google.auth import default

auth.authenticate_user()
creds, _ = default()
gc = gspread.authorize(creds)

worksheet = gc.open('DL-Ex1-DS').sheet1
data = worksheet.get_all_values()

dataset1 = pd.DataFrame(data[1:], columns=data[0])
dataset1 = dataset1.astype({'Input':'float'})
dataset1 = dataset1.astype({'Output':'float'})

dataset1.head()

X = dataset1[['Input']].values
y = dataset1[['Output']].values

X

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.33,random_state = 33)

Scaler = MinMaxScaler()
Scaler.fit(X_train)

X_train1 = Scaler.transform(X_train)

x_n = Scaler.fit_transform(X)

x_train,x_test,y_train,y_test = train_test_split(x_n,y,test_size = 0.3,random_state = 3)

ai = Sequential([
    Dense(8,activation = 'relu',input_shape=[1]),
    Dense(15,activation = 'relu'),
    Dense(1),
])

ai.compile(optimizer = 'rmsprop',loss = 'mse')

ai.fit(x_train,y_train,epochs=2000)
ai.fit(x_train,y_train,epochs=2000)

loss_plot = pd.DataFrame(ai.history.history)
loss_plot.plot()

ai.evaluate(x_test,y_test)

X_test1 = Scaler.transform(X_test)
X_n1 = [[30]]
X_n1_1 = Scaler.transform(X_n1)
ai.predict(X_n1_1)
```

## Dataset Information
![ss1](./img1.png)

## OUTPUT

### Training Loss Vs Iteration Plot

![ss2](./img2pt1.png)
![ss3](./img2pt2.png)

### Test Data Root Mean Squared Error
![ss4](./img3.png)

### New Sample Data Prediction
![ss5](./img4.png)
## RESULT

Thus a neural network regression model for the given dataset is written and executed successfully.
