![cover](https://images.pexels.com/photos/2988232/pexels-photo-2988232.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1)

# Credit Card Fraud

In this project, I'll predict if a credit card transaction is a fraud or not. I'll use [Kaggle Credit Card DB](https://www.kaggle.com/datasets/kartik2112/fraud-detection?resource=download), a synthetic generated data for USA clients from Jan/2019 to Jul/2020.

Different from other projects, this will only have [one notebook](https://github.com/Ana-Akaishi/data-science-projects/blob/main/Credit_card_fraud/CC_fraud.ipynb). The main difference is I'll be using only **PySpark** to load, treat, prepare and run ML models.

As usual, fraud projects like this have an issue of umbalanced dataset. Which means that fraud examples (y=1) are low compared to the total of observations. Since this projects simulates a real **Big Data** problem, I'll treat this issue by increasing the penalization for wrong classifications during the training phase.
- On previous [projects](https://github.com/Ana-Akaishi/data-science-projects/blob/main/Bank_Loan_Default/04%20-%20Comp%20Results.ipynb), I tested two different types of treatments for umbalanced datasets, and increasing penalization provided a better result.

## What models are you using?
Since this is a classification problem, I'll compare the classic and the most popular model:
- Logistical Regression (logit)
- Random Forest
- Gradient Boosting Trees: Equivalent of XGBoost, but in PySpark lib

## What variables should I look out for?
Usually, credit card fraud has a target group. So our fist step is to determin the **profile** of most frauded clients (sex, age, ocuppation), then we move to the **location**. With location we can narrow the profile to regional clients profiles, it's important to access the company risk and how many clients could be victim of fraud.

The tricky part of location is how to treat the data. Usually, you'll have *State*, *city* and *lot, long*. And here you have many ways to approach the data, you could create a dummy for each State and city, but this would increase the error and not really provide much since it's a binary variable. **I apporached location by using latitude and longitute (both float)**, this way I can later print it in a map and have a precise visualization of where the fraud occured, plus I can compare the distance in lat,long between the client and the marchant (who accepted the credit card transition). This provide an extra insight to identify a suspicious merchant.

When it comes to **feature engineering**, I did the classic time related:
- Month
- Year
- Day of week
- Week Number

### Why pick the 'week number'?
At first glance, week number seems like a random variable without much value to add. But since credit cards are used for purchase (digital or physical), we need to think **when do we use credit cards**? Retail industry has some important dates with many deals and increase in sales. For example, **Black Friday** has deals during the week of the event, and customers are expected to receive e-mails with offers, ads and other digital media with 'once in a lifetime deals'. This increases the possibilities for scams, phising and stealing credit card data.

So, during special holiday I should expect an increase in the probability of fraud and scams. But not all holidays are the same, for instance we cannot compare Black Friday deals with Easter. It's a different mix of product, price and public. So, I'll classify events in:
- Tier 1 event (Black Friday and Xmas): important holidays for retail and ecommerce. 
- Tier 2 event (Easter, 4th of July, Labor day, Halloween): holiday with some offers.
- No event: regular week.

#### Why not the day for the event?
Before the internet, retail used to give deals during the day to attract more people to stores. But today, the industry shifted to scatter the deals during the week so they don't overwork logistics and warehouses. Take Amazon for example, Black Friday deals start at the first week of November, by doing this they can still sell product with great discounts and leave only the big hits for the actual BF date.

By using the week, we catch all the opportunies for scams and frauds.