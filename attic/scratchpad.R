library(mlr)
devtools::load_all()
lrn = makeLearner("classif.xgboost", predict.type = "prob", nrounds = 10L)
mod = train(learner = lrn, task = iris.task)
task = iris.task
measures = list(acc, timetrain)
n_classes = length(task$task.desc$class.levels)


# Compute a set of different thresholds and nrounds
ncomb = ceiling(1000^(1 / n_classes))
threshold_vals = mlrMBO:::combWithSum(ncomb, n_classes) / ncomb
if (n_classes > 2) threshold_vals = rbind(threshold_vals, 1 / n_classes)
colnames(threshold_vals) = task$task.desc$class.levels
nrounds_vals  = quantile(seq_len(mod$learner$par.vals$nrounds), type = 1)


# Outer product over thresholds and nrounds
grd = expand.grid(i = seq_len(length(nrounds_vals)), j = seq_len(nrow(threshold_vals)))
out = Map(function(i, j) {list(nrounds = nrounds_vals[i], threshold = threshold_vals[j, ])},
  i = grd$i, j = grd$j)

# Compute performances
perfs = lapply(out, function(rw) {
  pp = predict_classif_with_subevals(mod, .task = task, ntreelimit = rw$nrounds, predict.threshold = rw$threshold)
  performance(pp, model = mod, measures = measures)
})

# Reduce to data.frame
list(
  y = convertListOfRowsToDataFrame(perfs),
  x = do.call("rbind", lapply(out, function(x) c(setNames(x$nrounds, "nrounds"), x$threshold)))
)
