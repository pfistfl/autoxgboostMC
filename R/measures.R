### Fairness Measures:
# I) Independence

#' Absolute differences of Positive Rate between groups
#' @export
fairpr = mlr::makeMeasure(id = "fairness.pr", minimize = TRUE, properties = c("classif", "response", "req.task"),
  extra.args = list(), best = 0, worst = 1,
  fun = function(task, model, pred, feats, extra.args) {
    groups = get_grouping(extra.args, 2L)
    fs = sapply(split(pred$data, f = groups), function(x) {
     mean(x$response == pred$task.desc$positive, na.rm = TRUE)
    })
    abs(max(fs) - min(fs))
  }
)

## 2) Sufficiency

#' Variance of F1 Scores between groups
#' @export
varf1 = mlr::makeMeasure(id = "fairness.varf1", minimize = TRUE, properties = c("classif", "response", "req.task"),
  extra.args = list(), best = 0, worst = 1,
  fun = function(task, model, pred, feats, extra.args) {

    if (is.character(extra.args$grouping)) {
      pred$data$groups = assert_factor(getTaskData(task)[[extra.args$grouping]]) # Task-column that is a factor
    } else if (is.function(extra.args$grouping)) {
      pred$data$groups = assert_factor(extra.args$grouping(getTaskData(task))) # Function that returns a factor
    } else {
      pred$data$groups = assert_factor(extra.args$grouping) # Or a factor.
    }
    fs = sapply(split(pred$data, f = pred$data$groups), function(x) {
     measureF1(x$truth, x$response, pred$task.desc$positive)
    })
    var(fs)
  }
)

#' Absolute differences of F1 Scores between groups
#' @export
fairf1 = mlr::makeMeasure(id = "fairness.f1", minimize = TRUE, properties = c("classif", "response", "req.task"),
  extra.args = list(), best = 0, worst = 1,
  fun = function(task, model, pred, feats, extra.args) {

    if (is.character(extra.args$grouping)) {
      pred$data$groups = assert_factor(getTaskData(task)[[extra.args$grouping]]) # Task-column that is a factor
    } else if (is.function(extra.args$grouping)) {
      pred$data$groups = assert_factor(extra.args$grouping(getTaskData(task))) # Function that returns a factor
    } else {
      pred$data$groups = assert_factor(extra.args$grouping) # Or a factor.
    }
    assert(length(levels(pred$data$groups)) == 2L)
    fs = sapply(split(pred$data, f = pred$data$groups), function(x) {
     measureF1(x$truth, x$response, pred$task.desc$positive)
    })
    abs(max(fs) - min(fs))
  }
)


## 3) Calibration
#' Absolute differences of Positive Rate between groups
#' @export
fairppv = mlr::makeMeasure(id = "fairness.ppv", minimize = TRUE, properties = c("classif", "response", "req.task"),
  extra.args = list(), best = 0, worst = 1,
  fun = function(task, model, pred, feats, extra.args) {
    groups = get_grouping(extra.args, 2L)
    fs = sapply(split(pred$data, f = groups), function(x) {
      measurePPV(x$truth, x$response, pred$task.desc$positive)
    })
    abs(max(fs) - min(fs))
  }
)

### Interpretability Measures:
# FIXME: Implement Christoph's IML Measures.

# #' Sparsity as percentage of used features
# #' @export
# xgb.sparsity = mlr::makeMeasure(id = "sparse.xgb", minimize = TRUE, properties = c("classif", "response"),
#   extra.args = list(), best = 0, worst = 1L,
#   fun = function(task, model, pred, feats, extra.args) {
#     if(model$learner$package != "xgboost") stop("Measure sparse.xgb is only available for xgboost models")
#     if(is.null(model$learner$par.vals$booster)) {
#       nfeats = length(setdiff(unique(xgboost::xgb.model.dt.tree(model = model$learner.model)$Feature), "Leaf"))
#       pctfeats = nfeats / sum(model$task.desc$n.feat)
#     } else {
#       pctfeats = 1
#     }
#     return(pctfeats)
#   }
# )


### Robustness Measures:

# Noise Injection
# Adversarial Examples





# Helper fucnctions
get_grouping = function(extra_args, n_levels = NULL) {
  if (is.character(extra.args$grouping)) {
    groups = assert_factor(getTaskData(task)[[extra.args$grouping]]) # Task-column that is a factor
  } else if (is.function(extra.args$grouping)) {
    groups = assert_factor(extra.args$grouping(getTaskData(task))) # Function that returns a factor
  } else {
    groups = assert_factor(extra.args$grouping) # Or a factor.
  }
  if (!is.null(n_levels)) assert_true(length(levels(pred$data$groups)) == n_levels)
  return(groups)
}
