# Machine Learning Car Price Prediction

The aim of this project is to introduce to the basic concept behind machine learning.

## Project Overview

In this project, i learned how to create a program that predicts the price of a car by using a linear function trained with a gradient descent algorithm. This is a fundamental machine learning concept, and it serves as a great starting point for anyone looking to dive into the world of data science and predictive modeling.

understanding of the following key components:
- Linear regression for predictive modeling
- Gradient descent algorithm for model training
- Data preprocessing and feature engineering
- Model evaluation and performance metrics

## Getting Started

To launch this project, you need follow these steps:

1. Clone the repository to your local machine:

   ```shell
   git clone https://github.com/ychibani/__linear_regression
  
2. Download the packages
   
   ```shell
   python3 -m pip install numpy matplotlib pandas
   
This project includes a sample dataset for car prices that you can use for training and testing your machine learning model. However, the knowledge and skills you gain from this project can be applied to any other dataset, allowing you to make predictions on various types of data.

3. launch the programs :

   ```shell
   python train.py -g
   ```
Train the model, print a graph of the gradient descent algorithm, ask the number of time we train the model as an input and calculate the thetas that will be sent in thetas.csv for predictions.py

![testle](https://github.com/ychibani42/__linear_regression/assets/55283897/7f8e6fad-081a-4912-9a09-986c3072f927)

   ```shell
   python predictions.py -g <price or data to predict>
   ```
Ask a mileage or a data input you want to predict with the values given in data.csv, get the thetas from thetas.csv and print a graph, and print a percentage of how relevant is my model, and the prediction

![youpi](https://github.com/ychibani42/__linear_regression/assets/55283897/3fd5ed06-9f44-4bd1-9d60-de690616b527)

   
