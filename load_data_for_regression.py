import numpy as np

file = '/content/airfoil_self_noise.csv'
# valid_file = 'data/kin8nm/validation.csv'
# test_file = 'data/kin8nm/test.csv'

print "loading data..."

train = np.loadtxt( open( file ), delimiter = "," )
# valid = np.loadtxt( open( valid_file ), delimiter = "," )
#test = np.loadtxt( open( test_file ), delimiter = "," )

y_train = train[:,-1]
y_test = train[:,-1]
#y_test = test[:,-1]

x_train = train[:,0:-1]
x_test = train[:,0:-1]
#x_test = test[:,0:-1]

data = { 'x_train': x_train, 'y_train': y_train, 'x_test': x_test, 'y_test': y_test }
