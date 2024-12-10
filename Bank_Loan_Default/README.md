![Alt textt](https://cdn.pixabay.com/photo/2019/09/27/17/02/rupee-4508945_1280.jpg)

# Bank Loan Default

For this project, I'll be focusing on using ML to identify potential client's default on their loans. For this, I'm going to use [Deloitte Hackathon](https://www.kaggle.com/datasets/ankitkalauni/bank-loan-defaulter-prediction-hackathon/data?select=test.csv) dataset, which accounts for a series of loans in India.

The dataset already comes split into train and test sets. The main issue with **default** prediction is that the dataset tends to be imbalanced. And what does that mean?
- In a clustering problem, we train our model using features (variables) and their corresponding 'answer'. In this case, if the client payed their loan or if they didn't (default)
- The ammount of clients that don't pay their loans is much smaller than the amount of clients that pay their debts. This makes our model bad at identifying 'bad clients'.

*Why the amount of default clients is smaller? Wouldn't we fix it if we picked older data as well?*
Default clients are a **tail event**. They are rare, otherwise the bank would go bankrupt. If we expand the timewindow we are using, we are just adding more noise into the model and the issue still there. You can't cherry pick default clients to make your dataset balanced. Every client is under a social economic condition **in that specific moment in time**. 

Basically, if you cherry pick default clients from past datasets (even from the same bank) you are adding unknonwn variables from that time period. And this will make a bigger mess and the model won't perform well.

## How to fix unbalanced datasets?
