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

- Additional Resources

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








