import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# file = '/content/airfoil_self_noise.csv'
# valid_file = 'data/kin8nm/validation.csv'
# test_file = 'data/kin8nm/test.csv'

data_train = pd.read_csv('/content/mnist_train.csv')

X = data_train.drop('label',axis=1) # Independet variable
y = data_train['label'] # Dependent variable

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=23)

data = { 'x_train': X_train, 'y_train': y_train, 'x_test': X_test, 'y_test': y_test }
