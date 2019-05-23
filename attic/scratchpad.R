library(mlr)
devtools::load_all()
lrn = makeLearner("classif.xgboost.custom", predict.type = "prob", nrounds = 10L)
mod = train(learner = lrn, task = iris.task)
pred = predict(mod, iris.task, ntreelimit = 3)
pred2 = predict(mod, iris.task, ntreelimit = 10)


pred
pred2
predict(mod$learner.model, num_trees = 10)