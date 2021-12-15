from math import sqrt
from sklearn.metrics import roc_auc_score as AUC, log_loss, accuracy_score as accuracy
from sklearn.metrics import mean_squared_error as MSE, mean_absolute_error as MAE, r2_score as r2
from hyperopt import hp as HP
from hyperopt.pyll.stochastic import sample
from ..base_model import BaseModel


class BaseRegressionModel(BaseModel):
    hp = HP

    def __init__(self, data, space):
        self.data = data
        self.__space = space

    def train_and_eval_model(self, model):
        x_train = self.data['x_train']
        y_train = self.data['y_train']

        x_test = self.data['x_test']
        y_test = self.data['y_test']

        model.fit( x_train, y_train )
        p = model.predict( x_train)

        mse = MSE(y_train, p)
        rmse = sqrt(mse)
        mae = MAE(y_train, p)
        r2_score = r2(y_train, p)

        print(("\n# training | RMSE: {:.4f}, MAE: {:.4f}, R2: {:.4f}".format( rmse, mae, r2_score )))


        p = model.predict(x_test)

        mse = MSE(y_test, p)
        rmse = sqrt(mse)
        mae = MAE(y_test, p)
        r2_score = r2(y_test, p)

        print(("# testing  | RMSE: {:.4f}, MAE: {:.4f}, R2: {:.4f}".format( rmse, mae, r2_score )))

        return { 'loss': rmse, 'rmse': rmse, 'mae': mae, 'r2': r2_score}

    def get_params(self):
        params = sample(self.__space)
        return self.handle_integers(params)

    def try_params(n_iterations, params):
        pass
