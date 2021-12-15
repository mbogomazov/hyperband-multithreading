"function (and parameter space) definitions for hyperband"
"binary classification with random forest"

from pprint import pprint
from sklearn.ensemble import RandomForestClassifier as RF
from base_classification_model import BaseClassificationModel

class HBRandomForestClassifier(BaseClassificationModel):
	trees_per_iteration = 5

	def __init__(self, data):
		self.__space = {
			'criterion': self.hp.choice( 'c', ( 'gini', 'entropy' )),
			'bootstrap': self.hp.choice( 'b', ( True, False )),
			'class_weight': self.hp.choice( 'cw', ( 'balanced', 'balanced_subsample', None )),
			'max_depth': self.hp.quniform( 'md', 2, 10, 1 ),
			'max_features': self.hp.choice( 'mf', ( 'sqrt', 'log2', None )),
			'min_samples_split': self.hp.quniform( 'msp', 2, 20, 1 ),
			'min_samples_leaf':self. hp.quniform( 'msl', 1, 10, 1 ),
		}
		self.data = data
		super().__init__(data, self.__space)

	def try_params(self, n_iterations, params):
		n_estimators = int(round(n_iterations * self.trees_per_iteration))
		# print(("n_estimators:", n_estimators))
		# pprint(params)
		model = RF( n_estimators = n_estimators, verbose = 0, n_jobs = -1, **params )
		return self.train_and_eval_model(model)
