# Bayesian Classification

## Project Overview
This project is a homework assigment for Advanced Machine Learning university course. It implements three Bayesian classification methods from scratch:
- **Linear Discriminant Analysis (LDA)**
- **Quadratic Discriminant Analysis (QDA)**
- **Naive Bayes (NB)**

The methods are evaluated on both simulated and real-world datasets to compare their performance.

### Simulated Data Experiments
Two different data generation schemes were used:
1. **Scheme 1**: Features for class 0 follow a standard normal distribution, while class 1 features have a shifted mean of `a`.
2. **Scheme 2**: Features follow a bivariate normal distribution with correlation `\rho`, where class 0 has a correlation of `\rho` and class 1 has `-\rho`.

For both schemes, experiments were conducted by:
- Varying `a` (`0.1, 0.5, 1, 2, 3, 5`) while fixing `rho = 0.5`
- Varying `rho` (`0, 0.1, 0.3, 0.5, 0.7, 0.9`) while fixing `a = 2`
- Repeating experiments with different train/test splits
- Generating **boxplots** to visualize accuracy distributions


### Real Data Experiments
- Used 4 datasets from [OpenML](https://www.openml.org/) - wine, MagicTelescope, pol, chscase_geyser1
- Datasets focus on binary classification with numerical features
- Repeating experiments with different train/test splits
- Generating **boxplots** to visualize accuracy distributions

