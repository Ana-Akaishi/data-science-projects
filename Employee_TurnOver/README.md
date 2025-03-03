![cover](https://images.pexels.com/photos/140945/pexels-photo-140945.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2)

# Employee Turn Over
This project objective is to create an algorithm to predict when an employee will leave the company and **identify the root cause** for it. This will be a **classification problem**, using the **[data set provided by Jiril Knsel](https://github.com/jiriknesl/employee-longevity)**.

## Datasets
I'll use the `data.example.csv` to train and test the models, for later apply it to `current_employee.example.csv` simulating a production state.

Different from fraud cases, where we'll always have an unbalanced class division, employee turn over may not have the same behavior. This needs to be investigated during the EDA phase. In case it's unbalanced, I'll treat it by increasing the model penalty for missclassification.

## Model
My model should focus on a binary classification: 1 if employee stay in the company and 0 otherwise. Later I can exercise creating a score to measure the probability to leave the company. This will give *human resources (HR)* input to act in retaining the employee.

For this, I'll explore the classics ML models and the hot gradient boosting trees algorithms:

- **Logistic Regression**
- **XGBoost** 
- **Light GBM**
- **CAT Boost**
- **Support Vectorial Machines**
- **Neural Networks** 

And finally check models performance using the Area Under the Curve (AUC). This metric roughtly translate as the model probability to classify a new observation right. My target is to get **AUC of 0.90**.

## Project Steps
Since this is a kaggle dataset and is already divided by train and test sets, there's not much cleaning or extra extractions to be made.
Instead, I'll jump right into our exploratory data analysis (EDA) and run some models. I'll deal with the inbalanced classes in two different ways, generating synthetic data and penalizing the model, each one in a different notebook:

00 - EDA

01 - Feature Engineering/Data preparation

02 - ML models

03 - Comparing Results

04 - Identifying key features
