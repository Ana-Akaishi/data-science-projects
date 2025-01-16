![alt text](https://images.unsplash.com/photo-1548668486-b554d9d443d4?q=80&w=1932&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D)

# Telecom Churn

In this project, I'll make a prediction model for telecom customer churn. The main difference in this project is that I'm going to set it as a real company project, which means it'll have the full production cycle:

1. Setting a database server with **Docker**: I'll load a public database into a container, so it simulates a SQL server to extract data.
    - Creating containers
    - Managing users and accesses
    - SQL language
2. Exploratory Data Analysis (EDA): analyze basic statistic and data structure
    - Data type
    - Null values
    - Feature engineering
    - Statistics (mean, outliers, boxplot, correlation)
3. Pediction Model Development: create a model to predict if a client will cancel their telecom contract or not
    - What model to use?
    - Why this model?
4. Deploy Model: create an API to predict new data entries

## What is churn?
[Churn](https://www.investopedia.com/terms/c/churnrate.asp) is another term for 'client turn over'. It calculates how many clients stopped doing business with the company ofter a while. Operations and management should analyze this metric along side with growth rate, so you could have a clear picture of the 'net growth'.

[Telecommunication industry](https://www.investopedia.com/ask/answers/070815/what-telecommunications-sector.asp) pays a lot of attention to **churn rate** due the high competition and product similarities (mobile lines, internet, covorage etc). The main weakness of churn rate is the lack of clarity for why the client left, so companies should try to prevent it by keeping their operations and price attractive.

Due to churn rate weakness, I'll focus on predicting **CHURN SCORE**. Which will give the company a probability score for each client to leave the company. This is a very powerful tool, since will indicate which clients they should focus special offers or reach out to lower their churn score. This will save company's money on publicity and achquiring new clients.

- Churn Score: a score from 0 to 100, where 0 means that the client have zero probability to leave and 100 that they will definitely leave.

## Dataset
I'll use a public dataset from [Hugging Face](https://huggingface.co/datasets/aai510-group1/telco-customer-churn). This is a simulated dataset derived from a a real telecom dataset, and has a training set, validation set and a testing set.

It's important to disclosure that this model **cannot be used to predict real churn in telecom industry**. This is just a project to exercise a full data science workflow. The results will be specific for this case and can't be generalized.

If you want to find more datasets like this one check [Dataset Search by Google](https://datasetsearch.research.google.com).

## Folder Structure
Since this projects simulates a prediction model production, I'll create a folder for each step: docker, model notebook, deployment.

### Docker - Database & SQL settings

```
0_sql_container/
├── db_admin/
│   └── 01_create_users.sql
├── load/
│   └── load_csv.py
├── docker/
│   └── docker-composer.yml
└── datasets/
    ├── train.csv
    ├── validation.csv
    └── test.csv
```

**How are you uploading your csv files to SQL database?**

I could create the tables by manually creating the tables with an .sql file, but since my datasets have 49 columns it's too much work. Instead, I'm going to use a python script to load it. Then I can transfer the SQL databse (with train, validation and test set) to my container and use it in my project!