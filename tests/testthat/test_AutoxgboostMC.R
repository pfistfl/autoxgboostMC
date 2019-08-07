context("AutoxgboostMC")

# test_that("autoxgboostMC works on different tasks for single measure",  {

#   if (EXPENSIVE_TESTS) {
#   tasks = list(
#     sonar.task, # binary classification
#     iris.fac,   # binary classification with factors
#     iris.task   # multiclass classification
#     # subsetTask(bh.task, subset = 1:50)
#     )
#   } else {
#     tasks = list(iris.task)
#   }

#   for (t in tasks) {
#     axgb = AutoxgboostMC$new(t)
#     expect_class(axgb, "R6")
#     axgb$fit(time_budget = 5L, plot = FALSE)
#     expect_class(axgb$opt_result, "MBOSingleObjResult")
#     expect_class(axgb$final_learner, "Learner")
#     expect_class(axgb$final_model, "WrappedModel")
#     p = axgb$predict(t)
#     expect_class(p, "Prediction")
#   }
# })

context("Multicrit")
test_that("Multiple measures work",  {
  library(mlr)
  fairf11 = setMeasurePars(fairf1, grouping = function(df) as.factor(df$age > 30))
  axgb = AutoxgboostMC$new(pid.task, measures = list(acc, fairf11))
  expect_class(axgb, "R6")
  if (EXPENSIVE_TESTS) {
    expect_warning(axgb$fit(time_budget = 5L, plot = FALSE))
    expect_class(axgb$optimizer$opt_result, "MBOResult")
    expect_class(axgb$final_learner, "Learner")
    axgb$fit_final_model()
    expect_class(axgb$final_model, "WrappedModel")
    p = axgb$predict(pid.task)
    expect_class(p, "Prediction")
  }
})

context("Printer")
test_that("autoxgboost printer works", {
  library(mlr)
  mod = AutoxgboostMC$new(pid.task, measures = list(auc))
  mod$fit(time_budget = 6L, plot = FALSE)
  expect_output(print(mod), "AutoxgboostMC tuning result")
  expect_output(print(mod), "Recommended parameters:")
  expect_output(print(mod), "eta:")
  expect_output(print(mod), "gamma:")
  expect_output(print(mod), "max_depth:")
  expect_output(print(mod), "colsample_bytree:")
  expect_output(print(mod), "colsample_bylevel:")
  expect_output(print(mod), "lambda:")
  expect_output(print(mod), "alpha:")
  expect_output(print(mod), "subsample:")
  # expect_output(print(mod), "nrounds:")
})
