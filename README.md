# autoxgboostMC - Multiple-Criteria tuning and fitting of [xgboost](https://github.com/dmlc/xgboost) models.

<img align="right" src="https://raw.githubusercontent.com/ja-thomas/autoxgboost/master/man/figures/hexagon.svg?sanitize=true" width="125px">


[![Build Status](https://travis-ci.org/pfistfl/autoxgboostMC.svg?branch=master)](https://travis-ci.org/pfistfl/autoxgboostMC)
[![Coverage Status](https://coveralls.io/repos/github/pfistfl/autoxgboostMC/badge.svg?branch=master)](https://coveralls.io/github/pfistfl/autoxgboostMC?branch=master)


* Installing the development version

    ```splus
    # Install requirements
    install.packages("devtools")
    devtools::install_github("compstat-lmu/paper_2019_iml_measures")
    devtools::install_github("johnmyleswhite/log4r")
    devtools::install_github("mlr-org/mlrMBO")
    # Install package
    devtools::install_github("ja-thomas/autoxgboostMC")
    ```

# General overview

autoxgboost aims to find an optimal [xgboost](https://github.com/dmlc/xgboost) model automatically using the machine learning framework [mlr](https://github.com/mlr-org/mlr)
and the bayesian optimization framework [mlrMBO](https://github.com/mlr-org/mlrMBO).

**Work in progress**!

AutoxgboostMC embraces `R6` for a cleaner design.
See the example code below for the new *API*.


First we split our data into train and test.
```r
train = sample(c(TRUE, FALSE), getTaskSize(pid.task), replace = TRUE)
task_train = subsetTask(pid.task, subset = train)
task_test = subsetTask(pid.task, subset = !train)
```

# Training and Testing
Then we start the AutoML process:

```r
# Instantiate the object with a list of measures to optimize.
axgb = AutoxgboostMC$new(task_train, measures = list(auc, timepredict))
# Set hyperparameters (we want to work on two threads
axgb$nthread(2L)
# Fit for 5 seconds
axgb$fit(time_budget = 15L)
```
after searching and finding a good model, we can use the best model to predict.

```r

p = axgb$predict(task_test)
```

## Visualizing the Process

```r
axgb$plot_opt_path()
axgb$plot_opt_result()
axgb$plot_pareto_front()
```


## Pipeline

AutoxgboostMC currently searches and optimizes the following Pipeline:

```r
fix_factors %>% impact_encoding | dummy encoding %>% drop_constant_feats %>% learner %>% tune_threshold
```

To be added:
- Categorical Encoding using mixed models
- Imputation

Eventually:
- Ensemble Stacking
- Model Compression

# autoxgboost - How to Cite

The **Automatic Gradient Boosting** framework was presented at the [ICML/IJCAI-ECAI 2018 AutoML Workshop](https://sites.google.com/site/automl2018icml/accepted-papers) ([poster](poster_2018.pdf)).
Please cite our [ICML AutoML workshop paper on arxiv](https://arxiv.org/abs/1807.03873v2).
You can get citation info via `citation("autoxgboost")` or copy the following BibTex entry:

```bibtex
@inproceedings{autoxgboost,
  title={Automatic Gradient Boosting},
  author={Thomas, Janek and Coors, Stefan and Bischl, Bernd},
  booktitle={International Workshop on Automatic Machine Learning at ICML},
  year={2018}
}
```
