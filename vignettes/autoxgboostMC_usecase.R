library(OpenML)
library(autoxgboostMC)
adult = convertOMLDataSetToMlr(getOMLDataSet(1590))
data = getTaskData(adult)
sex_fairf1 = setMeasurePars(fairf1, grouping = data$sex)

axgb = AutoxgboostMC$new(
  task = adult, # adult task
  measures = list(mmce, sex_fairf1) # measures to optimize
)

set.seed(20190606)
axgb$fit(iterations = 20L, plot = TRUE)

p1 = axgb$plot_pareto_front_projections(wt_range = c(0.1, 0.9))

axgb$optimizer$set_possible_projections(c(0.1, 0.9))
axgb$fit(iterations = 50L, plot = TRUE)
p2 = axgb$plot_pareto_front_projections(wt_range = c(0.1, 0.9))


axgb$optimizer$set_possible_projections(c(0.1, 0.9))
axgb$fit(iterations = 50L, plot = TRUE)
p3 = axgb$plot_pareto_front_projections(wt_range = c(0.1, 0.9))

library(patchwork)
library(ggplot2)
p = (p1  + coord_cartesian(xlim = c(0.1, 0.3), ylim = c(0, 0.03))) +
(p2  + coord_cartesian(xlim = c(0.1, 0.3), ylim = c(0, 0.03))) +
(p3  + coord_cartesian(xlim = c(0.1, 0.3), ylim = c(0, 0.03)))

# ggsave("pareto1.pdf")
# save(axgb, p1, p2, p3, file = "axgb.RData")




### Holdout autoxgboostMC
library(autoxgboostMC)
library(OpenML)
adult = convertOMLDataSetToMlr(getOMLDataSet(1590))
data = getTaskData(adult)
sex_fairf1 = setMeasurePars(autoxgboostMC::fairf1, grouping = data$sex)

n = adult$task.desc$size
set.seed(20190807)
train.ids = sample(1:n, 2/3 * n)
test.ids = setdiff(1:n, train.ids)

train.task = subsetTask(adult, train.ids)
test.task = subsetTask(adult, test.ids)


axgb = AutoxgboostMC$new(
  task = train.task, # adult task
  measures = list(mmce, sex_fairf1) # measures to optimize
)

set.seed(20190808)
axgb$fit(iterations = 120L, plot = TRUE)
library(mlrCPO)
build_model = function(pars, self) {
  pars = as.data.frame(t(pars))
  pars = trafoValue(self$parset, as.list(pars))
  pars = pars[!BBmisc::vlapply(pars, is.na)]
  threshold = pars$threshold
  pars$threshold = NULL
  lrn = makeLearner("classif.xgboost.custom", nrounds = pars$nrounds,
    objective = self$baselearner$par.vals$objective,
    predict.type = "prob", predict.threshold = threshold)
  lrn = setHyperPars2(lrn, par.vals = pars)
  lrn = self$pipeline$preproc_pipeline %>>% lrn
  return(lrn)
}

library(dplyr)
op = as.data.frame(getOptPathParetoFront(mlrMBO:::getOptStateOptPath(axgb$optimizer$opt_state)))
axgb$plot_pareto_front()

ids = as.numeric(rownames(op)[!duplicated(op)])
opt_parset = axgb$get_opt_path_df()[ids, 1:11]

lrns = apply(opt_parset, 1, build_model, self = axgb)

preds = lapply(lrns, FUN = function(x) {predict(train(x, train.task), task = test.task)})
perfs_mmce = unlist(lapply(preds, performance))
perfs_f1 = unlist(lapply(preds, FUN = function(x) {fairf1$fun(task = test.task, pred = x, extra.args = list(grouping = getTaskData(test.task)$sex))}))
axgb$plot_opt_path()

perf_df = data.frame(mmce = perfs_mmce, "fairF1" = perfs_f1)
perf_df$Method = rep("Pareto (ours)", length(perfs_mmce))
rownames(perf_df) = NULL
perf_df = perf_df %>%
  arrange(mmce, fairF1)
knitr::kable(perf_df[, c(3, 1, 2)], format = "latex")

### Holdout autoxgboost
set.seed(20190808)
autom = autoxgboost::autoxgboost(task = train.task, iterations=120L, build.final.model = TRUE)
fmod = autom$final.model
pred = predict(fmod, test.task)

performance(pred)
fairf1$fun(task = test.task, pred = pred, extra.args = list(grouping = getTaskData(test.task)$sex))
