# Hyperband Multithreading
Implementation of hyperband library with multithreading and ability to backup and restore state

## Installing
```bash
pip install hyperband-multithreading
```

## Usage

### Simple example of using Gradient Boosting classification model
```python
import pandas as pd
from hyperband_multithreading.hyperband import Hyperband
from hyperband_multithreading.models.classification.gb import HBGradientBoostingClassifier

data_train = pd.read_csv('mnist_train.csv')
X = data_train.drop('label',axis=1) # Independet variable
y = data_train['label'] # Dependent variable

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=23)
y_train
y_train_fixed = np.empty((y_train.shape[0], 10))
for ind, digit_ind in enumerate(y_train.values):
    new_row = np.zeros(10)
    new_row[digit_ind] = 1
    y_train_fixed[ind] = new_row

y_test_fixed = np.empty((y_test.shape[0], 10))
for ind, digit_ind in enumerate(y_test.values):
    new_row = np.zeros(10)
    new_row[digit_ind] = 1
    y_test_fixed[ind] = new_row

rf_data = { 'x_train': X_train, 'y_train': y_train,
            'y_train_acc': y_train_fixed, 'x_test': X_test,
            'y_test': y_test,
            'y_test_acc': y_test_fixed}


gb = HBGradientBoostingClassifier(rf_data)
hb_gb = Hyperband(gb)
results = hb_gb.run('backup_gb.back')
print(results)
```

### How to implement your own classification model
1. Place `model_name.py` to `models/classification` folder
2. Use this template to implement class for your model
```python
from models.base_classification_model import BaseClassificationModel

class HBYourModelNameClassifier(BaseClassificationModel):
	trees_per_iteration = <int>

	def __init__(self, data):
		self.__space = {}
		self.data = data
		super().__init__(data, self.__space)


	def try_params(self, n_iterations, params ):
		n_estimators = int(round(n_iterations * self.trees_per_iteration))
		model = your_model(n_estimators = n_estimators, verbose = 0, **params)
		return self.train_and_eval_model(model)
```
3. Import your model in `models/classification/__init__.py` file

## Todo
- [ ] Add ability to stop process when target value is equal or more than necessary
- [ ] Add ability to provide custom comparator
- [ ] Add more classification models implementation
- [ ] Add regression models implementation
- [ ] Add docstring to all files
- [ ] Refactor and lint code
- [ ] Add tests