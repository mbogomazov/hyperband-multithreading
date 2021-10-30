import numpy as np
import pandas as pd

file = '/content/airfoil_self_noise.csv'
# valid_file = 'data/kin8nm/validation.csv'
# test_file = 'data/kin8nm/test.csv'

data_2 = pd.read_csv('/content/airfoil_self_noise.csv',sep='\t')

X = data_2.drop('L',axis=1) # Independet variable
y = data_2['L'] # Dependent variable



data = { 'x_train': X, 'y_train': y, 'x_test': X, 'y_test': y }
