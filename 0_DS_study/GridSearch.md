# GridSearch

## What is GridSearch
In simple words, grid search is an algorithm that will try every combination of hyperparameters in your model and pick the best. Of course it comes with a high computational cost, but part of data science is experimenting with hyperparameters so you would have to do it eventually.

The idea is that you'll create a dictionary with all the **hyperparameters for that model** and the python library will do the rest for you. This means you still have to study and know the model so you can fine tune it.

EACH MODEL WILL HAVE A UNIQUE AND DIFFERENT HYPERPARAMETER DICTIONARY. So check the documentation of each model or my cheat sheet below (hehe).

## Python Library
I use [scikitlearn](https://scikit-learn.org/1.5/modules/generated/sklearn.model_selection.GridSearchCV.html). In python you'll use:

 `from sklearn.model_selection import GridSearchCV`

## Table of Contents
1. [Linear Regression](#linearR)
2. [Logistic Regression](#logit)
3. [Stochastic Gradient Descent](#sgd)
    1. [Regressor](#sgdr)
    2. [Classifier](##sgdc)
4. [Random Forest](#rf)
    1. [Regressor](#rfr)
    2. [Classifier](#rfc)
5. [XGBooost](#xgb)
6. [Light GBM](#lgbm)
7. [CatBoost](catb)
8. [K-Nearest Neighbors](#knn)
9. [Support Vector Machine]((#svm))
10. [Artificial Neural Networks](#ann)

## Hyperparameters
### Linear Regression <a name="linearR"></a>
```
from sklearn.linear_model import LinearRegression

lr_model = LinearRegression(*, fit_intercept=True, copy_X=True, n_jobs=None, positive=False)
```
Notice the only hyperparameter in this model is the **INTERCEPT**. 
The model will try to find the best intercept by default, so there's **NO GRIDSEARCH** to be done.

 ### Logistic Regression (Logit) <a name="logit"></a>
```
from sklearn.linear_model import LogisticRegression

logit_model = LogisticRegression(penalty='l2', *, dual=False, tol=0.0001, C=1.0, 
                                fit_intercept=True, intercept_scaling=1, 
                                class_weight=None, random_state=None, 
                                solver='lbfgs', max_iter=100, multi_class='deprecated', 
                                verbose=0, warm_start=False, n_jobs=None, l1_ratio=None)
```
In logit, we can play with some parameters:
- penalty: how the model will inflict penalties for wrong predictions (L1, L2 or elasticnet)
- C: corrects multicollinearity and overfitting in the model. The higher C value, less regularization is applied.
- solver: optimization algorithm ('lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga')

I make the dictionary this way:
```
param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],
    'penalty': ['l2', 'l1', 'elasticnet', 'none'],
    'solver': ['liblinear', 'lbfgs', 'saga']}
```

Some people also tune the parameter `max_iter`, however, parameter will be [deprecated](https://scikit-learn.org/1.5/modules/generated/sklearn.linear_model.LogisticRegression.html) in verson 1.5.
So I won't add it in this guide so it won't cause issues if someone else tries to use it.

### Stochastic Gradient Descent (SGD) <a name="sgd"></a>
This linear model can be used in **regressions** and **classifications**.

```
from sklearn.linear_model import SGDClassifier,SGDRegressor
```

#### SGD Regression <a name="sgdr"></a>
```
sgd_reg = SGDRegressor(loss='squared_error', *,
                        penalty='l2', alpha=0.0001, l1_ratio=0.15, 
                        fit_intercept=True, max_iter=1000, tol=0.001, 
                        shuffle=True, verbose=0, epsilon=0.1, random_state=None, 
                        learning_rate='invscaling', eta0=0.01, power_t=0.25, 
                        early_stopping=False, validation_fraction=0.1, 
                        n_iter_no_change=5, warm_start=False, average=False)
```
In regression, I can experiment with:
- loss: which loss function will be uses ('squared_error', 'huber', 'epsilon_insensitive', or 'squared_epsilon_insensitive')
- penalty: penalty applied ('l2', 'l1', 'elasticnet', 'none')
- alpha: regularization applied, and uses float values
- learning_rate: string for how the model will adjust its learning rate ('constant', 'optimal', 'invscaling', 'adaptive')
- eta0: learning rate and must be float

```
param_grid = {
    'loss': ['squared_error', 'huber', 'epsilon_insensitive'],
    'penalty': ['l2', 'l1', 'elasticnet'],
    'alpha': [0.0001, 0.001, 0.01],
    'learning_rate': ['constant', 'optimal', 'invscaling', 'adaptive'],
    'eta0': [0.001, 0.01, 0.1]
}
```

#### SGD Classifier <a name="sdgc"></a>
```
sgd_class = SGDClassifier(loss='hinge', *, 
                            penalty='l2', alpha=0.0001, l1_ratio=0.15, 
                            fit_intercept=True, max_iter=1000, tol=0.001, 
                            shuffle=True, verbose=0, epsilon=0.1, n_jobs=None,
                            random_state=None, learning_rate='optimal', 
                            eta0=0.0, power_t=0.5, early_stopping=False, 
                            validation_fraction=0.1, n_iter_no_change=5, 
                            class_weight=None, warm_start=False, average=False)
```
For classifier, we'll have the same parameters adding a few more:
- early_stopping: stop training if results don't get better (True, False)
- validation_fraction: activates if `early_stopping` is on, must be a float value.

```
param_grid = {
    'loss': ['hinge', 'log_loss', 'modified_huber', 'squared_hinge'],
    'penalty': ['l2', 'l1', 'elasticnet'],
    'alpha': [0.0001, 0.001, 0.01],
    'learning_rate': ['constant', 'optimal', 'invscaling', 'adaptive'],
    'eta0': [0.001, 0.01, 0.1],
    'early_stopping': [True, False]
}
```

### Random Forest <a name="rf"></a>
Random Forest can be used to **Regression** and **Classification**.

```
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
```
#### Random Forest Regressor <a name="rfr"></a>
```
rf_reg = RandomForestRegressor(n_estimators=100, *, 
                                criterion='squared_error', max_depth=None, 
                                min_samples_split=2, min_samples_leaf=1, 
                                min_weight_fraction_leaf=0.0, max_features=1.0, 
                                max_leaf_nodes=None, min_impurity_decrease=0.0, 
                                bootstrap=True, oob_score=False, n_jobs=None, 
                                random_state=None, verbose=0, warm_start=False, 
                                ccp_alpha=0.0, max_samples=None, monotonic_cst=None)
```
I'll change the following on regressors:
- n_estimators: # of trees
- criterion: criteria to split the tree (“squared_error”, “absolute_error”, “friedman_mse”, “poisson”)
- max_features: how many features the model will consider before spliting ('auto','sqrt','log2', 1.0)
- max_depth: how far the tree will go, if no value passed it'll split until reach the max # of leaf, must be int.
- ccp_alpha: reduce tree complexity by setting a complexity score, helps avoid overfitting.
- bootstrap: determine if the sample will be refreshed after spliting the branch.

notes:
- Tree complexity it means how far will spread

```
param_grid = {
    'n_estimators': [100, 200, 500],
    'max_depth': [10, 20, 50, None],
    'max_features': ['sqrt', 'log2', None],
    'bootstrap': [True, False],
    'ccp_alpha': [0.0, 0.01, 0.1]
}
```
#### Random Forest Classifier <a name="rfc"></a>
```
rf_clas = RandomForestClassifier(n_estimators=100, *, 
                                criterion='gini', max_depth=None, min_samples_split=2, 
                                min_samples_leaf=1, min_weight_fraction_leaf=0.0, 
                                max_features='sqrt', max_leaf_nodes=None, min_impurity_decrease=0.0, 
                                bootstrap=True, oob_score=False, n_jobs=None, random_state=None, 
                                verbose=0, warm_start=False, class_weight=None, 
                                ccp_alpha=0.0, max_samples=None, monotonic_cst=None)
```
For classifiers I'll change:
- n_estimators
- criterion: function to measure loss (“gini”, “entropy”, “log_loss”)
- max_depth
- bootstrap
- class_weight: sometimes the # of observations in each class isn't even and this will make your dataset imbalanced. For this case, you can turn this feature 'balanced'.
- ccp_alpha

```
param_grid = {
    'n_estimators': [100, 200, 500],
    'criterion': ['gini', 'entropy'],
    'max_depth': [10, 20, 50, None],
    'max_features': ['sqrt', 'log2', 'log_loss', None],
    'bootstrap': [True, False],
    'class_weight': [None, 'balanced']
}
```
### XGBooost (XGB) <a name="xgb"></a>

### Light GBM <a name="lgbm"></a>


### CatBoost - Ideal for CATegorical model <a name="catb"></a>

### K-Nearest Neighbors (KNN) <a name="knn"></a>

### Support Vector Machine (SVM) <a name="svm"></a>

### Artificial Neural Networks (ANN) <a name="ann"></a>