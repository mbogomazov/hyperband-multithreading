"function (and parameter space) definitions for hyperband"
"binary classification with XGBoost"

from numpy import log
from pprint import pprint
from xgboost import XGBClassifier as XGB
from base_classification_model import BaseClassificationModel

class HBXGBoostClassification(BaseClassificationModel):
	trees_per_iteration = 5

	def __init__(self, data):
		self.__space = space = {
			'learning_rate': self.hp.choice( 'lr', [
				'default',
				self.hp.uniform( 'lr_', 0.01, 0.2 )
			]),
			'max_depth': self.hp.choice( 'md', [
				'default',
				self.hp.quniform( 'md_', 2, 10, 1 )
			]),
			'min_child_weight': self.hp.choice( 'mcw', [
				'default',
				self.hp.quniform( 'mcw_', 1, 10, 1 )
			]),

			'subsample': self.hp.choice( 'ss', [
				'default',
				self.hp.uniform( 'ss_', 0.5, 1.0 )
			]),
			'colsample_bytree': self.hp.choice( 'cbt', [
				'default',
				self.hp.uniform( 'cbt_', 0.5, 1.0 )
			]),
			'colsample_bylevel': self.hp.choice( 'cbl', [
				'default',
				self.hp.uniform( 'cbl_', 0.5, 1.0 )
			]),
			'gamma': self.hp.choice( 'g', [
				'default',
				self.hp.uniform( 'g_', 0, 1 )
			]),
			'reg_alpha': self.hp.choice( 'ra', [
				'default',
				self.hp.loguniform( 'ra_', log( 1e-10 ), log( 1 ))
			]),
			'reg_lambda': self.hp.choice( 'rl', [
				'default',
				self.hp.uniform( 'rl_', 0.1, 10 )
			]),
			'base_score': self.hp.choice( 'bs', [
				'default',
				self.hp.uniform( 'bs_', 0.1, 0.9 )
			]),
			# 'scale_pos_weight': self.hp.choice( 'spw', [
			# 	'default',
			# 	self.hp.uniform( 'spw', 0.1, 10 )
			# ])
			}
		self.data = data
		super().__init__(data, self.__space)


	def get_params(self):
		params = self.sample( self.space )
		params = { k: v for k, v in list(params.items()) if v != 'default' }
		return self.handle_integers(params)


	def try_params(self, n_iterations, params):
		n_estimators = int(round(n_iterations * self.trees_per_iteration))
		# print("n_estimators:", n_estimators)
		# pprint(params)
		model = XGB( n_estimators = n_estimators, nthread = -1, use_label_encoder=False, eval_metric='mlogloss', **params)
		return self.train_and_eval_model(model)

