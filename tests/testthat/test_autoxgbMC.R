context("AutoxgboostMC")

test_that("autoxgboostMC works on different tasks for single measure",  {
  tasks = list(
    sonar.task, # binary classification
    iris.fac,   # binary classification with factors
    iris.task   # multiclass classification
    # subsetTask(bh.task, subset = 1:50)
    )

  for (t in tasks) {
    axgb = AutoxgboostMC$new(t)
    expect_class(axgb, "R6")
    axgb$fit(time_budget = 2L, plot = FALSE)
    expect_class(axgb$opt_result, "MBOSingleObjResult")
    expect_class(axgb$final_learner, "Learner")
    expect_class(axgb$final_model, "WrappedModel")
    p = axgb$predict(t)
    expect_class(p, "Prediction")
  }
})

test_that("Multiple measures work",  {
    fairf11 = setMeasurePars(fairf1, grouping = function(df) as.factor(df$age > 30))
    axgb = AutoxgboostMC$new(pid.task, measures = list(acc, fairf11))
    expect_class(axgb, "R6")
    axgb$tune_threshold = FALSE
    expect_warning(axgb$fit(time_budget = 5L, plot = FALSE))
    expect_class(axgb$opt_result, "MBOResult")
    expect_class(axgb$final_learner, "Learner")
    expect_class(axgb$final_model, "WrappedModel")
    p = axgb$predict(pid.task)
    expect_class(p, "Prediction")
})



context("Printer")
test_that("autoxgboost printer works", {
  mod = AutoxgboostMC$new(pid.task, measures = list(auc))
  expect_warning(mod$fit(time_budget = 3L, plot = FALSE))
  expect_output(print(mod), "Autoxgboost tuning result")
  expect_output(print(mod), "Recommended parameters:")
  expect_output(print(mod), "eta:")
  expect_output(print(mod), "gamma:")
  expect_output(print(mod), "max_depth:")
  expect_output(print(mod), "colsample_bytree:")
  expect_output(print(mod), "colsample_bylevel:")
  expect_output(print(mod), "lambda:")
  expect_output(print(mod), "alpha:")
  expect_output(print(mod), "subsample:")
  expect_output(print(mod), "nrounds:")
})
