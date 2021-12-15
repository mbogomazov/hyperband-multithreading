"function (and parameter space) definitions for hyperband"
"binary classification with gradient boosting"

from pprint import pprint
from sklearn.ensemble import GradientBoostingClassifier as GB
from base_classification_model import BaseClassificationModel

class HBGradientBoostingClassifier(BaseClassificationModel):
	trees_per_iteration = 5

	def __init__(self, data):
		self.__space = {
			'learning_rate': self.hp.uniform( 'lr', 0.01, 0.2 ),
			'subsample': self.hp.uniform( 'ss', 0.8, 1.0 ),
			'max_depth': self.hp.quniform( 'md', 2, 10, 1 ),
			'max_features': self.hp.choice( 'mf', ( 'sqrt', 'log2', None )),
			'min_samples_leaf': self.hp.quniform( 'mss', 1, 10, 1 ),
			'min_samples_split': self.hp.quniform( 'mss', 2, 20, 1 )
		}
		self.data = data
		super().__init__(data, self.__space)


	def try_params(self, n_iterations, params ):
		n_estimators = int(round(n_iterations * self.trees_per_iteration))
		# print("n_estimators:", n_estimators)
		# pprint(params)
		model = GB(n_estimators = n_estimators, verbose = 0, **params)
		return self.train_and_eval_model(model)

