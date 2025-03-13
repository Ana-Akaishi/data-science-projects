![cover](https://images.pexels.com/photos/2119758/pexels-photo-2119758.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2)

# Marketing Campaign A/B Testing
For this project, I'll analyse marketing campaign effectiveness with **A/B Testing**. This is an statistical approach to analyze if there's difference after applying the treatment, or in this case, after marketing promotions.

I'll use the [public database](https://www.kaggle.com/datasets/chebotinaa/fast-food-marketing-campaign-ab-test) available on Kaggle. Here, I'll test 3 different promotions on a fast food restaurant. Let's assume the dataset is a pilot, and the marketing team wants the DS to analyze which campaign was effective and they should invest.

## Dataset
Usually, A/B tests are made between a control group and the treatment (A and B). But for this case, marketing ran 3 promotions at the same time, and wants to know which one is better. So instead of comparing performaces with business as usual, I'll be comparing 3 treatments.

The dataset isn't big, so I'll approach it as a the final sample. I won't need to run the ideal sample size for this project.

Before running any statistical test, it's important to **check for normality**. This will guide me towards which statistical test run.

## Project Steps
First, I'll analyze the dataset and how the variables behave around each promotion. After that, I'll run the AB Test and provide the results.

00_EDA

01_AB_Test