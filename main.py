from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn import svm
import numpy as np
from xgboost import XGBClassifier

# set up train test split
X, y = load_svmlight_file("a9a.txt")
seed = 6
test_size = 0.4
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)

# load final test dataset
X_final, y_final = load_svmlight_file('a9a.t')

# evaluate model on train test split of a9a
def train_model(model):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    predictions = [round(value) for value in y_pred]
    accuracy = accuracy_score(y_test, predictions)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))

# test model on final testing dataset a9a.t
def test_model(model):
    final_pred = model.predict(X_final)
    predictions = [round(value) for value in final_pred]
    accuracy = accuracy_score(y_final, predictions)
    print("Final Accuracy: %.2f%%" % (accuracy * 100.0))


# tuning for learning rate
def xgb_learning_rate_tuning(model, start, stop):
    # define a set of numbers to test
    learnRates = np.linspace(start, stop)
    print("Testing values: ", learnRates)
    # GridSearchCV defaults to K-fold with 5 splits unless it detects a classifier in which case it will use a Straitfield 5-fold
    crossVal_lr = GridSearchCV(estimator=model, param_grid=dict(learning_rate=learnRates))
    crossVal_lr.fit(X_train, y_train)
    print("Best Error: ", crossVal_lr.best_score_)
    best = crossVal_lr.best_estimator_.learning_rate
    print("Best learning rate value: ", best)
    return best



# tuning for n_estimators
def xgb_n_estimators_tuning(model, start, stop, iter):
    num_estimators = np.linspace(start, stop, num=iter, dtype=int)
    print("Testing values: ", num_estimators)
    pgrid = dict(n_estimators=num_estimators)
    crossVal_ne = GridSearchCV(estimator=model, param_grid=pgrid)
    crossVal_ne.fit(X_train, y_train)
    print("Best n_estimators error: ", crossVal_ne.best_score_)
    best = crossVal_ne.best_estimator_.n_estimators
    print("Best n_estimators value: ", best)
    return best



# tuning for max_depth
def xgb_max_depth_tuning(model, start, stop, iter):
    depths = np.linspace(start, stop,num=iter, dtype=int)
    print('Testing values: ', depths)
    crossVal_depth = GridSearchCV(estimator=model, param_grid=dict(max_depth=depths))
    crossVal_depth.fit(X_train, y_train)
    print("Best max_depth error: ", crossVal_depth.best_score_)
    best = crossVal_depth.best_estimator_.max_depth
    print("Best max_depth value: ", best)
    return best



# tuning for lambda
def xgb_lambda_tuning(model, start, stop):
    lams = np.linspace(start, stop)
    print('Testing values: ', lams)
    crossVal_lam = GridSearchCV(estimator=model, param_grid=dict(reg_lambda=lams))
    crossVal_lam.fit(X_train, y_train)
    print("Best lambda error: ", crossVal_lam.best_score_)
    best = crossVal_lam.best_estimator_.reg_lambda
    print("Best lambda value: ", best)
    return best




# tuning for missing
def xgb_missing_tuning(model, start, stop):
    mis = np.linspace(start, stop)
    print('Testing values: ', mis)
    cross_val_missing = GridSearchCV(estimator=model, param_grid=dict(missing=mis))
    cross_val_missing.fit(X_train, y_train)
    print("Best missing error: ", cross_val_missing.best_score_)
    best = cross_val_missing.best_estimator_.missing
    print("Best missing value: ", best)
    return best


# SVM Hyper-parameter Tuning
# Kernel and gamma tuning can generally just be run once at the start to identify the best type
# C tuning is most taxing, but also the function you must iterate over most to really improve your model

# kernel tuning
def svm_kernel_tuning(model):
    type = ['linear', 'poly', 'rbf', 'sigmoid']
    print("Testing: ", type)
    cross_val_kernel = GridSearchCV(estimator=model, param_grid=dict(kernel=type))
    cross_val_kernel.fit(X_train, y_train)
    best = cross_val_kernel.best_estimator_.kernel
    print("Best kernel error: ", cross_val_kernel.best_score_)
    print("Best kernel type: ", best)
    return best

# find c value -> WARNING: this is taxing
def svm_c_tuning(model, start, stop, iter):
    Cs = np.linspace(start,stop, num=iter)
    print("Testing: ", Cs)
    print(Cs)
    cross_val_C = GridSearchCV(estimator=model, param_grid=dict(C=Cs), n_jobs=4)
    cross_val_C.fit(X_train, y_train)
    best = cross_val_C.best_estimator_.C
    print("Best C error: ", cross_val_C.best_score_)
    print("Best C Value: ", best)
    return best

# decide between auto or scale gamma
def svm_gamma_tuning(model):
    type = ['scale', 'auto']
    print("Testing: ", type)
    cross_val_gamma = GridSearchCV(estimator=model, param_grid=dict(gamma=type))
    cross_val_gamma.fit(X_train, y_train)
    best = cross_val_gamma.best_estimator_.gamma
    print("Best Gamma Error: ", cross_val_gamma.best_score_)
    print("Best Gamma Type: ", best)
    return best

'''
xgb_model = XGBClassifier()

# parameter tuning - first pass
n_estimators = dict(n_estimators=xgb_n_estimators_tuning(xgb_model, 50, 450))
max_depth = dict(max_depth=xgb_max_depth_tuning(xgb_model, 1, 10))
learning_rate = dict(learning_rate=xgb_learning_rate_tuning(xgb_model, 0, 1))
reg_lambda = dict(reg_lambda=xgb_lambda_tuning(xgb_model, 1, 10))
missing = dict(missing=xgb_missing_tuning(xgb_model, 0, 2))
obj = dict(objective='binary:logistic')
# set values
xgb_model.set_params(**n_estimators, **max_depth, **learning_rate, **reg_lambda, **missing, **obj)
print(xgb_model)
# evaluate model
train_model(xgb_model)

# first pass results
# n_estimators = 50 error = 0.8451064134683071
# max_depth = 3 error = 0.8485872010424561
# learning_rate = 0.08163265306122448 error = 0.8475122598364901
# reg_lambda = 7.244897959183674 error = 0.8473586107291352
# missing = 0.0 error = 0.8420862195482008
# obj = binary:logistic
# Accuracy - 84.18%

'''

'''
# parameter tuning - second pass

# updated model
xgb_model = XGBClassifier(n_estimators=50, max_depth=3, learning_rate=0.08163265306122448, reg_lambda=7.244897959183674,
                          missing=0.0, objective='binary:logistic', eval_metric='logloss')
#this time im going to set the params in between each update
n_estimators = dict(n_estimators=xgb_n_estimators_tuning(xgb_model, 50, 450, 20))
xgb_model.set_params(**n_estimators)
max_depth = dict(max_depth=xgb_max_depth_tuning(xgb_model, 1, 10, 10))
xgb_model.set_params(**max_depth)
learning_rate = dict(learning_rate=xgb_learning_rate_tuning(xgb_model, 0, 1))
xgb_model.set_params(**learning_rate)
reg_lambda = dict(reg_lambda=xgb_lambda_tuning(xgb_model, 1, 10))
xgb_model.set_params(**reg_lambda)
missing = dict(missing=xgb_missing_tuning(xgb_model, 0, 2))
xgb_model.set_params(**missing)
obj = dict(objective='binary:logistic')
xgb_model.set_params(**obj)

print(xgb_model)
# evaluate model
train_model(xgb_model)

# second pass results
# n_estimator = 386 error = 0.8498669422308174
# max_depth = 3 error = 0.8498669422308174
# learning rate = 0.08163265306122448 error = 0.8498669422308174
# lambda = 7.244897959183674  error = 0.8498669422308174
# missing = 0.0 error = 0.8498669422308174
# obj = binary:logistic
# Accuracy = 84.84%
'''

'''
# parameter tuning - third pass -> checking smaller intervals for param values
xgb_model = XGBClassifier(n_estimators=386, max_depth=3, learning_rate=0.08163265306122448, reg_lambda=7.244897959183674,
                          missing=0.0, objective='binary:logistic', eval_metric='logloss')

n_estimators = dict(n_estimators=xgb_n_estimators_tuning(xgb_model, 350, 450, 20))
xgb_model.set_params(**n_estimators)
max_depth = dict(max_depth=xgb_max_depth_tuning(xgb_model, 1, 10, 10))
xgb_model.set_params(**max_depth)
learning_rate = dict(learning_rate=xgb_learning_rate_tuning(xgb_model, 0, 0.2))
xgb_model.set_params(**learning_rate)
reg_lambda = dict(reg_lambda=xgb_lambda_tuning(xgb_model, 5, 8))
xgb_model.set_params(**reg_lambda)
missing = dict(missing=xgb_missing_tuning(xgb_model, 0.01, 1.5))
xgb_model.set_params(**missing)

print(xgb_model)
# evaluate model
train_model(xgb_model)

# third pass results
# n estimators = 386 error = 0.8498669422308174
# max_depth = 3 error = 0.8498669422308174
# learning_rate = 0.0816326530612245 error = 0.8498669422308174 
# reg_lambda = 5.183673469387755 error = 0.8498668898355548
# missing = 0.01 error = 0.8498668898355548
# Accuracy = 84.91%
#test_model(xgb_model)
'''

'''
# support vector machine
svm_model = svm.SVC(kernel='linear',gamma='scale')

# check kernel types
#kernel = dict(kernel=svm_kernel_tuning(svm_model))
#svm_model.set_params(**kernel)

# check gamma type
#gamma = dict(gamma=svm_gamma_tuning(svm_model))
#svm_model.set_params(**gamma)

# find a C value -> this can be taxing depending on the range of values you want to test
# For this reason, I kept this tuning rather short, but you could extend it if you wish
C = dict(C=svm_c_tuning(svm_model, 1, 10, 10))
svm_model.set_params(**C)


print(svm_model)

# evaluate model
train_model(svm_model)

# first pass values
# kernel = linear error = 0.8463348989911029
# gamma = scale error = 0.8463348989911029
# C = error =
'''


