![bank loand default cover](https://images.unsplash.com/photo-1565515636369-57f6e9f5fe79?q=80&w=1705&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D)

# Bank Loan Default

For this project, I'll be focusing on using ML to identify potential client's default on their loans. For this, I'm going to use [Deloitte Hackathon](https://www.kaggle.com/datasets/ankitkalauni/bank-loan-defaulter-prediction-hackathon/data?select=test.csv) dataset, which accounts for a series of loans in India.

The dataset already comes split into train and test sets. The main issue with **default** prediction is that the dataset tends to be imbalanced. And what does that mean?
- In a clustering problem, we train our model using features (variables) and their corresponding 'answer'. In this case, if the client payed their loan or if they didn't (default)
- The ammount of clients that don't pay their loans is much smaller than the amount of clients that pay their debts. This makes our model bad at identifying 'bad clients'.

*Why the amount of default clients is smaller? Wouldn't we fix it if we picked older data as well?*

Default clients are a **tail event**. They are rare, otherwise the bank would go bankrupt. If we expand the timewindow we are using, we are just adding more noise into the model and the issue still there. You can't cherry pick default clients to make your dataset balanced. Every client is under a social economic condition **in that specific moment in time**. 

Basically, if you cherry pick default clients from past datasets (even from the same bank) you are adding unknonwn variables from that time period. And this will make a bigger mess and the model won't perform well.

## How to fix inbalanced datasets?
There are a few ways to deal with inbalanced datasets. Since default is a very rare event, we can:
- Generate synthetic data for minority class
- Increase the penalty if the model classify the minority class

If you want more information, [this](https://dataheadhunters.com/academy/how-to-build-a-business-credit-risk-model-in-python/) is a good reference reading.
Even though is abour credit risk, the principles are the same for default loans. Rare events, clientes with N number of features on which we want to predict bad clients.

## Model
Loan default is nothing but a classifier problem. I need to classify clients between defaulter and non-defaulter, so a two class problem.

This situation allow me to use **Logistic Regression** to classify and predict bad (default) clients.
I'll compare the results of this linear regression with other models such as:
- **XGBoost**, popular for it's precision and applicability 
- [**Light GBM**](https://www.kaggle.com/code/prashant111/lightgbm-classifier-in-python), an alternative for gradient boosting
- **Support Vectorial Machines**, non-linear models that can be used to classification as well
- **Neural Networks** 

And finally check who perform better due the variables available.

## Project Steps
Since this is a kaggle dataset and is already divided by train and test sets, there's not much cleaning or extra extractions to be made.
Instead, I'll jump right into our exploratory data analysis (EDA) and run some models. I'll deal with the inbalanced classes in two different ways, generating synthetic data and penalizing the model, each one in a different notebook:

00 - EDA

01 - Feature Engineering

02 - ML with synthetic data

03 - ML increasing model penalization

04 - Comparing Results