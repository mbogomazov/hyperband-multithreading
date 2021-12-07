import numpy as np
from math import sqrt
from sklearn.metrics import roc_auc_score as AUC, log_loss, accuracy_score as accuracy
from sklearn.metrics import mean_squared_error as MSE, mean_absolute_error as MAE, r2_score as r2
from hyperopt import hp
from hyperopt.pyll.stochastic import sample


class BaseModel(object):
    def handle_integers(self, params):
        new_params = {}
        for k, v in list(params.items()):
            if type( v ) == float and int( v ) == v:
                new_params[k] = int( v )
            else:
                new_params[k] = v

        return new_params