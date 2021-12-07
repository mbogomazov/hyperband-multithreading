"imports and definitions shared by various defs files"


# handle floats which should be integers
# works with flat params

def train_and_eval_sklearn_regressor(reg, data):
	x_train = data['x_train']
	y_train = data['y_train']

	x_test = data['x_test']
	y_test = data['y_test']

	reg.fit( x_train, y_train )
	p = reg.predict( x_train)

	mse = MSE(y_train, p)
	rmse = sqrt(mse)
	mae = MAE(y_train, p)
	r2_score = r2(y_train, p)

	print(("\n# training | RMSE: {:.4f}, MAE: {:.4f}, R2: {:.4f}".format( rmse, mae, r2_score )))


	p = reg.predict(x_test)

	mse = MSE(y_test, p)
	rmse = sqrt(mse)
	mae = MAE(y_test, p)
	r2_score = r2(y_test, p)

	print(("# testing  | RMSE: {:.4f}, MAE: {:.4f}, R2: {:.4f}".format( rmse, mae, r2_score )))

	return { 'loss': rmse, 'rmse': rmse, 'mae': mae, 'r2': r2_score}

