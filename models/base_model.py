import pandas as pd
from sklearn.model_selection import train_test_split


class BaseModel(object):
    def handle_integers(self, params):
        new_params = {}
        for k, v in list(params.items()):
            if type( v ) == float and int( v ) == v:
                new_params[k] = int( v )
            else:
                new_params[k] = v

        return new_params

    def load_and_split_data(filepath):
        data_train = pd.read_csv(filepath)
        X = data_train.drop('label',axis=1) # Independet variable
        y = data_train['label'] # Dependent variable
        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=23)
        return { 'x_train': X_train, 'y_train': y_train, 'x_test': X_test, 'y_test': y_test}