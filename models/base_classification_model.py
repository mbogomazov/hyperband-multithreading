from pandas import DataFrame
import numpy as np
from math import sqrt
from sklearn.metrics import roc_auc_score as AUC, log_loss, accuracy_score as accuracy
from sklearn.metrics import mean_squared_error as MSE, mean_absolute_error as MAE, r2_score as r2
from hyperopt import hp as HP
from hyperopt.pyll.stochastic import sample
from base_model import BaseModel


class BaseClassificationModel(BaseModel):
    hp = HP

    def __init__(self, data, space):
        self.data = data
        self.__space = space

    def train_and_eval_model(self, model):
        x_train = self.data['x_train']
        y_train = self.data['y_train']
        y_train_acc = self.data['y_train_acc']

        x_test = self.data['x_test']
        y_test = self.data['y_test']
        y_test_acc = self.data['y_test_acc']

        model.fit(x_train, y_train)

        # try:
        # 	p = clf.predict_proba( x_train )[:,1]	# sklearn convention
        # except IndexError:
        p = model.predict_proba(x_train)

        ll = log_loss(y_train, p)
        auc = AUC(y_train, p, multi_class='ovo')
        acc = accuracy(y_train_acc, np.round(p))

        # print(("\n# training | log loss: {:.2%}, AUC: {:.2%}, accuracy: {:.2%}".format( ll, auc, acc )))


        # try:
        # 	p = clf.predict_proba( x_test )[:,1]	# sklearn convention
        # except IndexError:
        p = model.predict_proba(x_test)

        ll = log_loss(y_test, p)
        auc = AUC(y_test, p, multi_class='ovo')
        acc = accuracy(y_test_acc, np.round(p))

        # print(("# testing  | log loss: {:.2%}, AUC: {:.2%}, accuracy: {:.2%}".format( ll, auc, acc )))

        return { 'loss': ll, 'log_loss': ll, 'auc': auc, 'acc': acc }

    def get_params(self):
        params = sample(self.__space)
        return self.handle_integers(params)

    def try_params(n_iterations, params):
        pass
