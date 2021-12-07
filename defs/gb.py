"function (and parameter space) definitions for hyperband"
"binary classification with gradient boosting"

from sklearn.ensemble import GradientBoostingClassifier as GB

from hyperband.common_defs import *
from hyperband.defs.base_classification_model import BaseClassificationModel

class HGradientBoostingClassifier(BaseClassificationModel):
	trees_per_iteration = 5

	# subsample is good for showing out-of-bag errors
	# when fitting in verbose mode, and probably not much else
	space = {
		'learning_rate': hp.uniform( 'lr', 0.01, 0.2 ),
		'subsample': hp.uniform( 'ss', 0.8, 1.0 ),
		'max_depth': hp.quniform( 'md', 2, 10, 1 ),
		'max_features': hp.choice( 'mf', ( 'sqrt', 'log2', None )),
		'min_samples_leaf': hp.quniform( 'mss', 1, 10, 1 ),
		'min_samples_split': hp.quniform( 'mss', 2, 20, 1 )
	}

	def __init__(self, data):
		super(BaseClassificationModel, self).__init__()
		self.data = data

	def get_params(self):
		params = sample( self.space )
		return handle_integers( params )

	def try_params(self, n_iterations, params ):
		n_estimators = int( round( n_iterations * self.trees_per_iteration ))
		print("n_estimators:", n_estimators)
		pprint( params )
		clf = GB( n_estimators = n_estimators, verbose = 0, **params )
		return train_and_eval_sklearn_classifier( clf, self.data )

