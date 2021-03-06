context("Measures")
test_that("Fairness measures work",  {
  library(mlr)
  fairf11 = setMeasurePars(fairf1, grouping = function(df) as.factor(df$age > 30))
  fairpr1 = setMeasurePars(fairpr, grouping = function(df) as.factor(df$age > 30))
  fairppv1 = setMeasurePars(fairppv, grouping = function(df) as.factor(df$age > 30))
  lrn = makeLearner("classif.rpart")
  mod = train(lrn, pid.task)
  prd = predict(mod, pid.task)
  expect_class(prd, "Prediction")
  perf = performance(prd, fairf11, task = pid.task)
  expect_numeric(perf, lower = 0, upper = 1)
  perf = performance(prd, fairpr1, task = pid.task)
  expect_numeric(perf, lower = 0, upper = 1)
  perf = performance(prd, fairppv1, task = pid.task)
  expect_numeric(perf, lower = 0, upper = 1)
})



context("Robustness")
test_that("Robustness measures",  {
  library(mlr)
  tasks = list(
  sonar.task, # binary classification
  iris.fac,   # binary classification with factors
  iris.task   # multiclass classification
  )
  for (t in tasks) {
    lrn = makeLearner("classif.rpart")
    mod = train(lrn, t)
    prd = predict(mod, t)
    expect_class(prd, "Prediction")
    perf = performance(prd, robustnoise, task = t, model = mod)
    expect_numeric(perf, lower = 0, upper = 1)
    perf = performance(prd, robustnoiseperfeat, task = t, model = mod)
    expect_numeric(perf, lower = 0, upper = 1)
  }
})

# FIXME: Include this when measures are faster.
context("Interpretability")
test_that("Robustness measures",  {
  library(mlr)
  lrn = makeLearner("classif.rpart", predict.type = "prob")
  task = subsetTask(pid.task, subset = sample(c(1:100, 500:600)), features = 1:2)
  mod = train(lrn, task)
  prd = predict(mod, task)
  expect_class(prd, "Prediction")
  perf = performance(prd, interpnf2, task = task, model = mod)
  expect_integerish(perf, lower = 0, upper = Inf)

  #   lrn = makeLearner("classif.rpart", predict.type = "prob")
  #   task = subsetTask(pid.task, subset = sample(c(1:100, 500:600)), features = 1:2)
  #   mod = train(lrn, task)
  #   prd = predict(mod, task)
  #   expect_class(prd, "Prediction")
  #   interpmec1 = setMeasurePars(interpmec, grid.size = 2L, max_seg_cat = 2L, max_seg_num = 2L, epsilon = 0.2)
  #   interpias1 = setMeasurePars(interpias, grid.size = 2L, max_seg_cat = 2L, max_seg_num = 2L, epsilon = 0.2)
  #   interpnf1 = setMeasurePars(interpnf,  grid.size = 2L, max_seg_cat = 2L, max_seg_num = 2L, epsilon = 0.2)
  #   perf = performance(prd, interpmec, task = task, model = mod)
  #   expect_numeric(perf, lower = 0, upper = Inf)
  #   perf = performance(prd, interpias, task = task, model = mod)
  #   expect_numeric(perf, lower = 0, upper = Inf)
  #   perf = performance(prd, interpnf, task = task, model = mod)
  #   expect_integerish(perf, lower = 0, upper = Inf)
})

