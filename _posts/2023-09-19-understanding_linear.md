---
date: 2023-09-19 22:55:45
layout: post
title: Understanding Linear Regression. Mathematical Foundations and Implementations in PyTorch and TensorFlow
image: https://i.ibb.co/6mQpjv1/logo.webp
optimized_image: https://i.ibb.co/6mQpjv1/logo.webp
category: Neural Network
tags:
  - machine learning
  - Neural Network
  - Pytorch
  - Tensorflow
  - Keras
  - Linear Rgression
author: shivangx01b
---

### Understanding Linear Regression. Mathematical Foundations and Implementations in PyTorch and TensorFlow
###### Blog Post by - Shivang Kumar

## Contents

- Introduction

- The Mathematics of Linear Regression
    - Definition
    - Equation
    - Cost Function
    - Optimization

- Implementing Linear Regression in PyTorch
    - Setting Up the Environment
    - Data Preparation
    - Model Definition
    - Loss and Optimizer
    - Training Loop
    
- Implementing Linear Regression in TensorFlow
    - Setting Up the Environment
    - Data Preparation
    - Model Definition
    - Loss and Optimizer
    - Training Loop
    
- Conclusion


## Introduction

**Linear regression** stands as one of the most fundamental algorithms in the realm of machine learning and predictive analytics. Rooted in statistical modeling, it serves as a cornerstone technique, particularly for those embarking on their data science journey. The beauty of linear regression lies in its simplicity and interpretability, making it an invaluable tool for making predictions based on relationships between numerical variables.

At its core, linear regression aims to establish a linear relationship between a **dependent variable** (often denoted as y) and one or more **independent variables** (denoted as x). The model predicts the outcome (y) as a linear combination of the input variables (x). This simplicity, however, belies its power and versatility in various applications, ranging from economics and healthcare to engineering and beyond.

In the upcoming sections, we will delve deep into the mathematical underpinnings of linear regression. Understanding these fundamentals is crucial for appreciating how the algorithm makes predictions and how it can be fine-tuned for better performance. We will explore the key concepts such as the linear regression equation, the cost function, and the optimization process.

Following this, we'll transition into the practical aspects of linear regression, showcasing its implementation in two of the most popular machine learning frameworks: **PyTorch** and **TensorFlow**. These sections will not only include detailed code examples but will also provide insights into the unique features and approaches of each framework. Whether you are a novice in the field or looking to refresh your knowledge, this comprehensive guide will equip you with both the theoretical understanding and practical skills to master linear regression.

Stay tuned as we embark on this exciting journey to unravel the intricacies of linear regression and harness its power in PyTorch and TensorFlow!. And will try to predict **House price** which Linear regression model

- House Price Chart

![Alt Text](https://i.ibb.co/n3BZ9TG/house.webp)


## The Mathematics of Linear Regression



Definition of Linear Regression

Linear regression is a statistical method used in machine learning and data analysis for modeling the relationship between a dependent variable and one or more independent variables. The essence of linear regression is to find a linear relationship or trend in the data. This approach is particularly useful for predicting the value of the dependent variable based on the values of the independent variables.

![Alt Text](https://i.ibb.co/Y7Z7tF3/lr.png)

The Linear Equation: y = mx + c
The fundamental equation of linear regression is y = mx + c. Here:

y represents the dependent variable - the variable we aim to predict or estimate.
x represents the independent variable - the variable we use for making predictions.
m is the slope of the line, which indicates the rate at which y changes for a unit change in x.
c is the y-intercept, the value of y when x is 0.
In the context of our house price example, if y represents house prices and x represents house sizes, m would indicate how much the price increases per square meter, and c would be the base price of a house when its size is zero.



Cost Function: Mean Squared Error (MSE)

![Alt Text](https://i.ibb.co/S39hpnb/mse.png)

The Mean Squared Error (MSE) is a widely used cost function for linear regression. It measures the average squared difference between the actual and predicted values. The formula for [MSE](https://en.wikipedia.org/wiki/Mean_squared_error) is:

**MSE = N 1 ​ ∑ i=1 N ​ (actual y ​ −predicted y ​ ) 2**

where N is the number of data points. The goal in linear regression is to minimize this MSE, which would imply that our model's predictions are as close to the actual values as possible.

Optimization: Gradient Descent
Gradient descent is an optimization algorithm used to minimize the MSE in linear regression. It works by iteratively adjusting the parameters m and c to reduce the cost (MSE). The algorithm takes steps proportional to the negative of the gradient (or approximate gradient) of the function at the current point. If correctly implemented, it leads us to the minimum of the function, i.e., the best values of m and c for our linear model.


## Implementing Linear Regression in PyTorch

- Setting Up the Environment
To get started with PyTorch, you need to install it in your Python environment. PyTorch can be installed via pip or conda, and you should choose the version compatible with your system and CUDA (if you're using GPU acceleration). For CPU-only installation, you can use the following pip command:

```bash
pip install torch torchvision
```
For specific installation instructions tailored to your system (including GPU support), visit the [PyTorch official website.](https://pytorch.org/get-started/locally/)

- Data Preparation
For this implementation, we'll create a simple dataset representing house prices predicted based on size, bedrooms, age of the house. We'll use synthetic data for this example:

```python
import torch

# Dummy data
# Features: size, bedrooms, age of the house
features = torch.tensor([[1200, 3, 20],
                         [1500, 4, 15],
                         [850, 2, 25],
                         [950, 2, 30],
                         [1700, 5, 10],
                         [1300, 3, 22]], dtype=torch.float32)

# Prices in $1000s
prices = torch.tensor([[200],
                       [250],
                       [180],
                       [190],
                       [300],
                       [220]], dtype=torch.float32)
```
- Model Definition
We'll define a simple linear regression model using torch.nn.Linear, which takes 3 inputs (size, bedrooms, age of the house) and give 1 output (Prices)

```python
# Creating a simple PyTorch model for predicting house prices
class LinearHouseModel(nn.Module):
    def __init__(self):
        super(LinearHouseModel, self).__init__()
        # Assuming house prices depend on 3 features
        self.linear = nn.Linear(3, 1)

    def forward(self, x):
        return self.linear(x)

# Creating a new model
mean = features.mean(0, keepdim=True)
std = features.std(0, unbiased=False, keepdim=True)
normalized_features = (features - mean) / std
model = LinearHouseModel()
```

- Loss and Optimizer
The Mean Squared Error (MSE) loss is used as it's the most common loss function for regression problems. We'll use the Stochastic Gradient Descent (SGD) optimizer:

```python
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
```

- Training Loop
The training process involves a forward pass to compute predictions and loss, followed by a backward pass to compute gradients and update the model parameters:

```python
# The model with the normalized data
num_epochs = 1000
train_losses = []
for epoch in range(num_epochs):
    # Forward pass
    outputs = model(normalized_features)
    loss = criterion(outputs, prices)

    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    train_losses.append(loss.item())
```
- Visualizing the Result
After training, you can plot the original data points and the model's predictions to visualize how well your model has learned:

```python
plt.plot(range(num_epochs), train_losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Over Time')
plt.legend()
plt.show()
```

![Alt Text](https://i.ibb.co/58TqcJZ/loss.png)

```python
# Predictions
predictions = model(normalized_features).detach().numpy()

# Plotting the actual and predicted prices
plt.scatter(np.arange(len(prices)), prices, label='Actual Prices')
plt.scatter(np.arange(len(predictions)), predictions, label='Predicted Prices', color='red')
plt.xlabel('House')
plt.ylabel('Price (in $1000s)')
plt.title('Actual vs Predicted House Prices')
plt.legend()
plt.show()
```

![Alt Text](https://i.ibb.co/3Wvc0Xb/preidict.png)


##  Implementing Linear Regression in TensorFlow

- Setting Up the Environment
To begin working with TensorFlow, you first need to install it. TensorFlow can be easily installed using pip. Ensure you have Python installed on your system, and then run the following command:

```bash
pip install tensorflow
```

- Data Preparation
For this implementation, we'll create a simple dataset representing house prices predicted based on size, bedrooms, age of the house. We'll use synthetic data for this example:

```python
# Dummy data
# Features: size, bedrooms, age of the house
features = np.array([[1200, 3, 20],
                         [1500, 4, 15],
                         [850, 2, 25],
                         [950, 2, 30],
                         [1700, 5, 10],
                         [1300, 3, 22]], dtype=float)

# Prices in $1000s
prices = np.array([[200],
                       [250],
                       [180],
                       [190],
                       [300],
                       [220]], dtype=float)
```

- Model Definition
In TensorFlow, we use the Keras API to define a linear model. Keras provides a simple and intuitive way to build models in TensorFlow:

```python
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

mean = features.mean(axis=0)
std = features.std(axis=0)
normalized_features = (features - mean) / std

model = keras.Sequential([
    keras.layers.Dense(1, input_shape=(3,))
```

- Loss and Optimizer
For the loss function, we'll use Mean Squared Error (MSE), and for the optimizer, we'll use Stochastic Gradient Descent (SGD):

```python
from tensorflow.keras.optimizers import SGD

sgd = SGD(learning_rate=0.01)
model.compile(optimizer=sgd, loss='mean_squared_error')
```

- Training Loop
Training the model in TensorFlow is streamlined using the fit method, which handles the training loop internally:

```
history = model.fit(normalized_features, prices, epochs=1000, verbose=0)
```

- Visualizing the Result
After training, you can plot the original data points and the model's predictions to visualize how well your model has learned:

```python
plt.plot(range(num_epochs), train_losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Over Time')
plt.legend()
plt.show()
```

![Alt Text](https://i.ibb.co/58TqcJZ/loss.png)

```python
# Predictions
predictions = model(normalized_features).detach().numpy()

# Plotting the actual and predicted prices
plt.scatter(np.arange(len(prices)), prices, label='Actual Prices')
plt.scatter(np.arange(len(predictions)), predictions, label='Predicted Prices', color='red')
plt.xlabel('House')
plt.ylabel('Price (in $1000s)')
plt.title('Actual vs Predicted House Prices')
plt.legend()
plt.show()
```

![Alt Text](https://i.ibb.co/3Wvc0Xb/preidict.png)


## Conclusion

Summarizing Key Points
In this comprehensive guide, we delved into the world of linear regression, one of the most fundamental algorithms in machine learning and predictive analytics. We started by exploring the mathematical foundations of linear regression, including the linear equation y = mx + c, the Mean Squared Error (MSE) as the cost function, and the gradient descent optimization method.

We then transitioned to practical implementations, demonstrating how to build linear regression models in two of the leading machine learning frameworks: PyTorch and TensorFlow. Each framework was explored through the lens of a simple yet illustrative example: predicting house prices based on house size.

Differences and Similarities in Implementation
While PyTorch and TensorFlow both effectively handle linear regression tasks, their approaches and syntax exhibit some differences:

PyTorch offers a more granular level of control with explicit training loops, making it a go-to choice for research and development where such flexibility is crucial.
TensorFlow, particularly with its Keras API, provides a more high-level, streamlined approach, making it user-friendly, especially for beginners and in production environments.
Despite these differences, both frameworks share core similarities, such as the use of tensors for data representation, backpropagation for learning, and a strong community support.





