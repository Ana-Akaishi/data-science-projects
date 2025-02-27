# Machine Learning Classification Models
The objective of this type of model is to classify our data based on known groups. It can be a binary classification (where the y = 0|1), or we can ask for the model to create probabilities.

The basic approach is the same for either situations, just the output and interpretation is different. For example, imagine we create a model to forecast the weather, it can be **binary** to predict if will rain (y=1) or not (y=0), or we can create a **probability** model. In this case, we'll have the probability of rain for the day. If we have more than one class, let's say cloudy, the model will provide the probability for each class.

## Models recommentation
- Logistic Regression: the classic
- Random Forest: another classic, worth the benchmark
- Naive Bayes: another classic, but **can't process negative data**
- K-means: last of the classic list, and for this **you have to know the number of classes in the dataset** (*maybe better for a sorting problem rather than binary class or probability*)
- Gradient Boosting Trees: most popular nowadays
    1. XGBoost
    2. Light GBM
    3. CatBoost
- Support Vector Machine (SVM): requires standardize data
- Artifical Neural Network (ANN)*: requires standardize data

*note: ANNs perform better if you have a lot of data. Else, chances are gradient boosting models will be better.*

## Main libs

```
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import lightgbm as lgbm
from catboost as cat
from sklearn import svm
from sklearn import metrics
from sklearn.inspection import partial_dependence
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD, Adam
```

## Unbalanced Datasets
Sometimes, our classes are not balanced in the dataset. This means our model will be good predicting the class with more observations because it's easier to get it right. To deal with this, we have two options:
1. Create synthetic data for minority class (SMOT)
2. Increase penalization for minority class missclassification

Academic paper usually tend to lean towards SMOT, but for real life projects I recommend increase penalty for missclassification (as my results on [Bank Load Default](https://github.com/Ana-Akaishi/data-science-projects/blob/main/Bank_Loan_Default/04%20-%20Comp%20Results.ipynb) project).

## Project steps
1. Check data quality
    - Null values
    - Type of data in column
    - Duplicates
    - Basic feature engineering (date related variables)
2. Exploratory Data Analysis (EDA): see this as answering business questions. This will give you insight to create new variables and understand the dataset.
    - Y class distribution
    - Feature/variable analysis
    - Statistical analysis
    - Outliers: I suggest do a boxplot
    - Correlation matrix
3. Data preparation
    - Transform string columns into int (dummy variables)
    - Drop name and datetime columns: most of models won't process strings or datetime formats
    - Feature engineering
    - Create minority class weight (*if your dataset is unbalanced*): # majority class/ # minority class
    - Split dataset in training, validation and test sets (recommended proportion: 70% train, 15% validation, 15% test)
    - Standardize your data: do this **after spliting your main dataset** between train, validation and test. The ideais to create a scaler based on the **train set**, and then use it to transform validation and test set. This way you won't cheat the model results.
4. Training model & Results
    - Train model
    - Predict using test set
    - Plot a confusion matrix
    - Analyze key metrics: Accuracy, Recall, Precision and F-1 Score (*if possible, always check F-1 since it's an avarage of the other metrics*)
    - Check Area Under the Curve (AUC): will provide the model probability the classify a new observation right. This metric account for issues in the dataset.
5. Pick a model
    - Model with hights AUC
    - Highest F-1 Score, checl the 'macro avarage F1 Score'
    - [Inspect features/variable impact](https://scikit-learn.org/stable/modules/partial_dependence.html): here you'll create an array with the coefficient for each feature. This will show how much each variable 'explain' or 'contribuited' to the model.