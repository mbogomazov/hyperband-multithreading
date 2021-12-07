"function (and parameter space) definitions for hyperband"
"binary classification with XGBoost"

from xgboost import XGBClassifier as XGB

#
from hyperband.common_defs import *

# a dict with x_train, y_train, x_test, y_test
# from hyperband.load_data import data
from hyperband.defs.base_classification_model import BaseClassificationModel

class HXGBoostClassification(BaseClassificationModel):
	trees_per_iteration = 5
	space = {
		'learning_rate': hp.choice( 'lr', [
			'default',
			hp.uniform( 'lr_', 0.01, 0.2 )
		]),
		'max_depth': hp.choice( 'md', [
			'default',
			hp.quniform( 'md_', 2, 10, 1 )
		]),
		'min_child_weight': hp.choice( 'mcw', [
			'default',
			hp.quniform( 'mcw_', 1, 10, 1 )
		]),

		'subsample': hp.choice( 'ss', [
			'default',
			hp.uniform( 'ss_', 0.5, 1.0 )
		]),
		'colsample_bytree': hp.choice( 'cbt', [
			'default',
			hp.uniform( 'cbt_', 0.5, 1.0 )
		]),
		'colsample_bylevel': hp.choice( 'cbl', [
			'default',
			hp.uniform( 'cbl_', 0.5, 1.0 )
		]),
		'gamma': hp.choice( 'g', [
			'default',
			hp.uniform( 'g_', 0, 1 )
		]),
		'reg_alpha': hp.choice( 'ra', [
			'default',
			hp.loguniform( 'ra_', log( 1e-10 ), log( 1 ))
		]),
		'reg_lambda': hp.choice( 'rl', [
			'default',
			hp.uniform( 'rl_', 0.1, 10 )
		]),
		'base_score': hp.choice( 'bs', [
			'default',
			hp.uniform( 'bs_', 0.1, 0.9 )
		]),
		# 'scale_pos_weight': hp.choice( 'spw', [
		# 	'default',
		# 	hp.uniform( 'spw', 0.1, 10 )
		# ])
	}

	def __init__(self, data):
		super(BaseClassificationModel, self).__init__()
		self.data = data

	def get_params(self):
		params = sample( self.space )
		params = { k: v for k, v in list(params.items()) if v != 'default' }
		return handle_integers( params )

	#

	def try_params(self, n_iterations, params, get_predictions = False ):
		n_estimators = int( round( n_iterations * self.trees_per_iteration ))
		print("n_estimators:", n_estimators)
		pprint(params)
		clf = XGB( n_estimators = n_estimators, nthread = -1, use_label_encoder=False, eval_metric='mlogloss', **params)
		return train_and_eval_sklearn_classifier( clf, self.data )

