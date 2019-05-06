context("AutoxgboostMC")

test_that("autoxgboostMC works on different tasksfor single measure",  {
  tasks = list(
    sonar.task, # binary classification
    iris.fac,   # binary classification with factors
    iris.task,  # multiclass classification
    subsetTask(bh.task, subset = 1:50),
    iris.fac)

  for (t in tasks) {
    axgb = AutoxgboostMC$new(measures = list(acc))
    axgb$fit(t, time.budget = 5L)
    expect_true(!is.null(axgb$model))
    p = axgb$predict(t)
    expect_class(p, "Prediction")
  }
})

test_that("autoxgboostMC works on different tasks",  {

  tasks = list(
    sonar.task, # binary classification
    iris.fac,   # binary classification with factors
    iris.task)  # multiclass classification

  for (t in tasks) {
    axgb = AutoxgboostMC$new(t, measures = list(acc, timepredict))
    axgb$fit(time.budget = 4L)
    expect_class(axgb$opt_result, "MBOMultiObjResult")
    p = axgb$predict(t)
    expect_class(p, "Prediction")
  }
})

test_that("New measures work",  {
    fairf11 = setMeasurePars(fairf1, grouping = function(df) as.factor(df$age > 30))
    axgb = AutoxgboostMC$new(pid.task, measures = list(acc, fairf11, timepredict))
    axgb$set_threshold_tuning(FALSE)
    axgb$fit(time.budget = 10L)
    expect_true(!is.null(axgb$model))
    p = axgb$predict(pid.task)
    expect_class(p, "Prediction")
})



