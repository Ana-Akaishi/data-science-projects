# A/B Testing and DS

[A/B testing](https://www.analyticsvidhya.com/blog/2020/10/ab-testing-data-science/) became very popular for being a simple statistical method to determine the difference between two groups (A and B). In a business sense, we can use A/B testing to determine the performance of marketing campaing, UI changes, promotions and so on.

Usually, these test are done with a sample, since the company will run a pilot to gather the data before compromising. After that, we will do **hypothesis thesis**, it's pretty similar how you did it for your undergrad graduation project:
    - Null Hypothesis (H0): there **IS NO** difference between groups
    - Hypothesis 1 (H1): there IS a difference between groups

Once the hypothesis are set, we need run some test and analyze their **statistical significance**. For this, I'll use as reference the P-Value.
- P-Value is interpreted as the possibility of the **null hypothesis** be true
- Threshold: test p-value < 0.05

It is important to note that A/B test only tells if there's a difference between groups without poiting preciselly why. We assume it's wathever event we are investigating (sales, new interface, campaing). For a more robust and statistical approach we should us **Difference in Differences**, an experimental method used by determine causal inference in many fields such as medice/health care and economics.

A/B test design:
1. Define hypothesis with business areas
    - what is the problem we want to solve? (increase sales, brand awareness, screen time)
    - what is our primary metric? (revenue, conversion, seconds on the page, # of clicks, # of views)
    - will it impact any KPI?
    - define hypothesis
2. A/B testing design and [Power Analysis](https://www.spotfire.com/glossary/what-is-power-analysis)
    - power analysis: probability of CORRECTLY REJECTING THE NULL HYPOTHESIS
    - usually, [statistical power runs about** 0.80**](https://github.com/renatofillinich/ab_test_guide_in_python/blob/master/AB%20testing%20with%20Python.ipynb)
    - minimum sample size
3. Run A/B test
4. Result Analysis (statistic significance)

Here's a [crash course](https://www.youtube.com/watch?v=KZe0C0Qq4p0) in case you like classes or more detailed explenation.

## What libraries to use
For Python, we'll use statsmodels and scipy:

```
import statsmodels.stats.api as sms     # Used to do the calculate the minimum sample size
from scipy.stats import shapiro, ttest_1samp, ttest_ind, levene, kruskal, mannwhitneyu, pearsonr, spearmanr
```

or for a generic overview `import scipy.stats as stats`

### Minimum Sample Size
The `main_metric_groupA` is our current benchmark, usually comes from business as usual metrics. While `main_metric_groupB` will be an ESTIMATION for the treatment expected result. There's two way to calculate the GROUP B metric:
1. Set a target for the main metric
2. Set the difference you want to see

Example, current conversion rate is 13% (main_metric_groupA = 0.13):
1. I want the new converstion rate to be 15% (main_metric_groupB = 0.15) since we'll add a Whatsapp button
2. I'm expecting an increase in 2% (0.13 + 0.02 = 0.15 -> main_metric_groupB = 0.15) since previous change in UX designed provided similar results

```
effect_size = sms.proportion_effectsize(main_metric_groupA, main_metric_groupB)    

required_n = sms.NormalIndPower().solve_power(
    effect_size, 
    power=0.8, 
    alpha=0.05, 
    ratio=1
    )                                                  # Calculating sample size needed

required_n = ceil(required_n)                          # Rounding up to next whole number                          

print(required_n)
```

### What are those test?
**shapiro (Shapiro Wilk)**
- Tests if the group has a normal distribution or not (H0: group has normal distribution)
- This is important because depending on the result, you'll use a different set of [statistical analysis](https://www.analyticsvidhya.com/blog/2021/06/hypothesis-testing-parametric-and-non-parametric-tests-in-statistics/) to compare group A and B
    - Group HAVE a normal distribution: use parametric statistics (T-Test, f_oneway, ANOVA)
    - Group DON'T HAVE a normal distribution: use nonparametric statistics (Mann-Whitney U Test, Kruskal)
- P-Value < 0.05, then the, then the **group DOES NOT have a normal distribution**

#### Parametric tests
**ttest_1samp ([T-test](https://www.jmp.com/en/statistics-knowledge-portal/t-test))**
- Compares mean between groups A and B
- It will give 2 results, the numeric difference between group means and the second is the **statistic significance** of it
- P-Value < 0.05, then the **means are different and we can reject the null hypothesis**

*obs: if you have compare sample and population, use `ttest_1samp`. If you are comparing sample A and B, use `ttest_ind`*

**personr (Person Correlation)**
- This is DOES NOT IMPLY CAUSALITY
- Lienar relationship between two groups
- P-value < 0.05, then **groups are correlated**

#### Nonparametric tests
**levene (Levene Test)**
- Compares if the [variance](chrome-extension://efaidnbmnnnibpcajpcglclefindmkaj/https://www.bgsu.edu/content/dam/BGSU/college-of-arts-and-sciences/center-for-family-and-demographic-research/documents/Help-Resources-and-Tools/Statistical%20Analysis/Annotated-Output-T-Test-SPSS.pdf) between groups is statistical significant
- The null hypothesis is: groups are have variance homogeneity (same distribution)
- P-value < 0.05, then **groups have different variance**

**kruskal (Kruskal-Wallis H-test or Non-parametric ANOVA test)**
- Compares if the median of group A and B are the same
- Null hypothesis (H0): A and B have the same median
- P-value < 0.05, then **groups have different medians**

**mannwhitneyu ([Mann-Whitney U test](https://datatab.net/tutorial/mann-whitney-u-test))**
- Compares two different samples central tendency
- [Central tendency](https://statistics.laerd.com/statistical-guides/measures-central-tendency-mean-mode-median.php) are statistical measures of mean, mode and median
- Null hypothesis (H0): There is no difference (in terms of central tendency) between the two groups in the population.
- P-value <0.05, then **groups have different central tendency**

*obs: if you want to run the same test but for more than 2 groups, run Kruskal-Wallis*

**spearmanr (Spearman rank-order correlation)**
- Nonparametric version of Pearson correlation
- You'll have to change parameter depending on sample size (n<500)
- P-value indicates if the correlation is statistic significant
- P-value < 0.05, then **groups have ordinal correlation**

https://www.kaggle.com/code/yunusemreturkoglu/ab-testing-anova/notebook