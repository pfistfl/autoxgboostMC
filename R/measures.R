#' Fairness Measures
#'
#' Fair algorithms usually are defined as having similar performances within
#' different dataset splits.
#' We implement the following measures:\cr
#'
#' Independence `fairpr` \cr
#' Sufficiency `fairf1` \cr
#' Callibration `fairppv` \cr
#'
#' The arguments can be set via `extra.args` for all measures.
#'
#' @param grouping [`function` | `factor` | `character`]\cr
#'   Either a function(df), a factor or a character column name that returns a factor.
#'   If `function`, df is the output of `getTaskData()`.
#'   This factor is used to split the data and compute differences between groups.
#' @rdname fairness_measures
#' @name fairness_measures
#' @examples \dontrun{setMeasurePars(fairf1, grouping = function(df) {as.factor(df$age > 30)})}
NULL

# 1) Independence
#' Absolute differences of Positive Rate between groups
#'
#' @export
fairpr = mlr::makeMeasure(id = "fairness.pr", minimize = TRUE, properties = c("classif", "response", "req.task"),
  extra.args = list(), best = 0, worst = 1,
  fun = function(task, model, pred, feats, extra.args) {
    groups = get_grouping(task, extra.args, 2L)
    fs = sapply(split(pred$data, f = groups), function(x) {
     mean(x$response == pred$task.desc$positive, na.rm = TRUE)
    })
    abs(max(fs) - min(fs))
  }
)

## 2) Sufficiency

#' Absolute differences of F1 Scores between groups
#' See Hardt et al, 2016: https://arxiv.org/pdf/1610.02413.pdf
#' @export
fairf1 = mlr::makeMeasure(id = "fairness.f1", minimize = TRUE, properties = c("classif", "response", "req.task"),
  extra.args = list(), best = 0, worst = 1,
  fun = function(task, model, pred, feats, extra.args) {
    groups = get_grouping(task, extra.args, 2L)
    fs = sapply(split(pred$data, f = groups), function(x) {
     measureF1(x$truth, x$response, pred$task.desc$positive)
    })
    abs(max(fs) - min(fs))
  }
)

#' Variance of F1 Scores between groups
#' @export
varf1 = mlr::makeMeasure(id = "fairness.varf1", minimize = TRUE, properties = c("classif", "response", "req.task"),
  extra.args = list(), best = 0, worst = 1,
  fun = function(task, model, pred, feats, extra.args) {
    groups = get_grouping(task, extra.args)
    fs = sapply(split(pred$data, f = groups), function(x) {
     measureF1(x$truth, x$response, pred$task.desc$positive)
    })
    var(fs)
  }
)




## 3) Calibration

#' Absolute differences of Positive Predictive Value between groups
#' @export
fairppv = mlr::makeMeasure(id = "fairness.ppv", minimize = TRUE, properties = c("classif", "response", "req.task"),
  extra.args = list(), best = 0, worst = 1,
  fun = function(task, model, pred, feats, extra.args) {
    groups = get_grouping(task, extra.args, 2L)
    fs = sapply(split(pred$data, f = groups), function(x) {
      measurePPV(x$truth, x$response, pred$task.desc$positive)
    })
    abs(max(fs) - min(fs))
  }
)

### Interpretability Measures:
# FIXME: We have to compute 'imeasure' globally only once. Or aggregate, otherwise this is very slow.

#' Interpretability Measures
#'
#'Several measures for interpretability have been defined in [Molnar et al., 2019](https://arxiv.org/abs/1904.03867).
#' We implement those measures the following measures:
#'
#' Curve Complexity `interpmec` \cr
#' Interaction Strength `interpias` \cr
#' Number of features `interpnf` \cr
#'
#' The arguments can be controlled via `extra.args` for the aforementioned measures.
#'
#' @param grid.size [`integer(1)`]\cr
#'   Controls the grid to evaluate on. Default: 10L.
#' @param max_seg_cat [`integer(1)`]\cr
#'   Max. number of segments for categorical variables. Default: 5L.
#' @param max_seg_num [`integer(1)`]\cr
#'   Max. number of segments for numerical variables. Default: 5L.
#' @param epsilon [`numeric(1)`]\cr
#'   1 - Minimum R^2.
#' @rdname interpretability_measures
#' @name interpretability_measures
#' @examples \dontrun{setMeasurePars(interpmec, grid.size = 10L)}
NULL

#' Curve Complexity
#' See `?interpretability_measures` for additonal info
#' For more info see Molnar et al., 2019 https://arxiv.org/abs/1904.03867
#' @export
interpmec = mlr::makeMeasure(id = "interp.mec", minimize = TRUE, properties = c("classif", "response", "req.task", "req.model"),
  extra.args = list(grid.size = 10, max_seg_cat = 5, max_seg_num = 5, epsilon = 0.05), best = 0, worst = 1,
  fun = function(task, model, pred, feats, extra.args) {
    grid.size = assert_integerish(extra.args$grid.size)
    max_seg_cat = assert_integerish(extra.args$max_seg_cat)
    max_seg_num = assert_integerish(extra.args$max_seg_num)
    epsilon = assert_numeric(extra.args$epsilon)

    task_data = getTaskData(task)
    pred = iml::Predictor$new(model, task_data, y = task$task.desc$target)
    pred$class = 1L
    imeasure = autoiml:::FunComplexity$new(pred, max_seg_cat = max_seg_cat, max_seg_num = max_seg_num, epsilon = epsilon, grid.size = grid.size)
  return(round(imeasure$c_wmean, 1))
  }
)

#' Interaction Strength
#' See `?interpretability_measures` for additonal info
#' For more info see Molnar et al., 2019 https://arxiv.org/abs/1904.03867
#' @export
interpias = mlr::makeMeasure(id = "interp.ias", minimize = TRUE, properties = c("classif", "response", "req.task", "req.model"),
  extra.args = list(grid.size = 10, max_seg_cat = 5, max_seg_num = 5, epsilon = 0.05), best = 0, worst = 1,
  fun = function(task, model, pred, feats, extra.args) {
    grid.size = assert_integerish(extra.args$grid.size)
    max_seg_cat = assert_integerish(extra.args$max_seg_cat)
    max_seg_num = assert_integerish(extra.args$max_seg_num)
    epsilon = assert_numeric(extra.args$epsilon)

    task_data = getTaskData(task)
    pred = iml::Predictor$new(model, task_data, y = task$task.desc$target)
    pred$class = 1L
    imeasure = autoiml:::FunComplexity$new(pred, max_seg_cat = max_seg_cat, max_seg_num = max_seg_num, epsilon = epsilon, grid.size = grid.size)
    return(imeasure$n_features)
  }
)

#' Number of features
#' See `?interpretability_measures` for additonal info
#' For more info see Molnar et al., 2019 https://arxiv.org/abs/1904.03867
#' @export
interpnf = mlr::makeMeasure(id = "interp.nfeat", minimize = TRUE, properties = c("classif", "response", "req.task", "req.model"),
  extra.args = list(grid.size = 10, max_seg_cat = 5, max_seg_num = 5, epsilon = 0.05), best = 0, worst = 1,
  fun = function(task, model, pred, feats, extra.args) {
    grid.size = assert_integerish(extra.args$grid.size)
    max_seg_cat = assert_integerish(extra.args$max_seg_cat)
    max_seg_num = assert_integerish(extra.args$max_seg_num)
    epsilon = assert_numeric(extra.args$epsilon)

    task_data = getTaskData(task)
    pred = iml::Predictor$new(model, task_data, y = task$task.desc$target)
    pred$class = 1L
    imeasure = autoiml:::FunComplexity$new(pred, max_seg_cat = max_seg_cat, max_seg_num = max_seg_num, epsilon = epsilon, grid.size = grid.size)
  return(imeasure$n_features)
  }
)
#' Number of features via noise injection
#'
#' Injects noise into each column of the prediction data.
#' This is done as follows:
#' For `numeric` features we add rnorm(0, diff(range(x)) * eps)
#' For `factors` we randomly flip eps % to a randomly selected factor level.
#' The predictions on original data and on corrupted data are then compared using
#' accuracy as a measure.
#' This corruption process is iterated for every feature and averaged.
#' @param eps [`numeric(1)`]\cr
#'   Magnitude of injected noise. Default: 0.01
#' @param n [`integer(1)`]\cr
#'   Repliactions for the corruption process. Results are averaged. Default: 1L.
#' @export
interpnf2 = mlr::makeMeasure(id = "interp.nfeat2", minimize = TRUE, properties = c("classif", "response", "req.task", "req.model"),
  extra.args = list(eps = 0.7, n = 1L), best = 0, worst = 1,
  fun = function(task, model, pred, feats, extra.args) {
    eps = assert_numeric(extra.args$eps)
    reps = assert_integerish(extra.args$n)
    # Repeat n times for robustness.
    res = vnapply(seq_len(extra.args$n), function(i) {
      perfeat_res = vnapply(seq_len(getTaskNFeats(task)), function(feature) {
        noise_pred = predict(model, newdata = inject_noise_single_feature(task, feature, eps))
        mean(pred$data$response == noise_pred$data$response, na.rm = TRUE)
      })
      sum(perfeat_res == 1)
    })
    mean(res)
  }
)

### Robustness Measures:

#' Robustness Measures
#'
#' We implement the following measures:
#'
#' Noise corruption on full dataset: `robustnoise` \cr
#' Featurewise corruption: `robustnoiseperfeat` \cr
#'
#' The following arguments can be controlled via `extra.args`:
#' @param eps [`numeric(1)`]\cr
#'   Magnitude of injected noise. Default: 0.01
#' @param n [`integer(1)`]\cr
#'   Repliactions for the corruption process. Results are averaged. Default: 1L.
#' @rdname robustness_measures
#' @name robustness_measures
#' @examples \dontrun{setMeasurePars(robustnoise, eps = 0.05)}
NULL

#' Noise Injection full dataset
#'
#' Injects noise into the prediction data. This is done as follows:
#' For `numeric` features we add rnorm(0, diff(range(x)) * eps)
#' For `factors` we randomly flip eps % to a randomly selected factor level
#' @param eps [`numeric(1)`]\cr
#'   Magnitude of injected noise. Default: 0.01
#' @param n [`integer(1)`]\cr
#'   Repliactions for the corruption process. Results are averaged. Default: 1L.
#' @export
robustnoise = mlr::makeMeasure(id = "robustness.noise", minimize = FALSE, properties = c("classif", "response", "req.task", "req.model"),
  extra.args = list(eps = 0.01, n = 5L), best = 0, worst = 1,
  fun = function(task, model, pred, feats, extra.args) {
    eps = assert_numeric(extra.args$eps)
    reps = assert_integerish(extra.args$n)
    # Repeat n times for robustness.
    res = vnapply(seq_len(extra.args$n), function(i) {
      noise_pred = predict(model, newdata = inject_noise_task(task, eps))
      mean(pred$data$response == noise_pred$data$response, na.rm = TRUE)
    })
    mean(res)
  }
)

#' Featurewise Noise Injection
#'
#' Injects noise into each column of the prediction data.
#' This is done as follows:
#' For `numeric` features we add rnorm(0, diff(range(x)) * eps)
#' For `factors` we randomly flip eps % to a randomly selected factor level.
#' The predictions on original data and on corrupted data are then compared using
#' accuracy as a measure.
#' This corruption process is iterated for every feature and averaged.
#' @param eps [`numeric(1)`]\cr
#'   Magnitude of injected noise. Default: 0.01
#' @param n [`integer(1)`]\cr
#'   Repliactions for the corruption process. Results are averaged. Default: 1L.
#' @export
robustnoiseperfeat = mlr::makeMeasure(id = "robustness.perfeat.noise", minimize = FALSE, properties = c("classif", "response", "req.task", "req.model"),
  extra.args = list(eps = 0.01, n = 5L), best = 0, worst = 1,
  fun = function(task, model, pred, feats, extra.args) {
    eps = assert_numeric(extra.args$eps)
    reps = assert_integerish(extra.args$n)
    # Repeat n times for robustness.
    res = vnapply(seq_len(extra.args$n), function(i) {
      perfeat_res = vnapply(seq_len(getTaskNFeats(task)), function(feature) {
        noise_pred = predict(model, newdata = inject_noise_single_feature(task, feature, eps))
        mean(pred$data$response == noise_pred$data$response, na.rm = TRUE)
      })
      mean(perfeat_res)
    })
    mean(res)
  }
)

# Adversarial Examples

# Distribution shift


### Helper functions

#' Obtain a grouping factor from either a data column, an additional factor or a function.
get_grouping = function(task, extra_args, n_levels = NULL) {
  if (is.character(extra_args$grouping)) {
    groups = assert_factor(getTaskData(task)[[extra_args$grouping]]) # Task-column that is a factor
  } else if (is.function(extra_args$grouping)) {
    groups = assert_factor(extra_args$grouping(getTaskData(task))) # Function that returns a factor
  } else {
    groups = assert_factor(extra_args$grouping) # Or a factor.
  }
  if (!is.null(n_levels)) assert_true(length(levels(groups)) == n_levels)
  return(groups)
}

#' Inject noise into every feature of a dataset.
inject_noise_task = function(task, eps = 0.01) {
  feats = getTaskData(task, target.extra = TRUE)$data
  feats = lapply(feats, inject_noise, eps = eps)
  return(data.frame(do.call("cbind", feats)))
}

#' Inject noise into a single feature of a dataset.
inject_noise_single_feature = function(task, feature, eps = 0.05) {
  assert_integerish(feature)
  feats = getTaskData(task, target.extra = TRUE)$data
  feats[, feature] = inject_noise(feats[, feature], eps = eps)
  return(data.frame(feats))
}

#' Inject noise into a single feature
inject_noise = function(x, eps) {
  if (is.numeric(x)) {
    # For numeric features we inject noise with rnorm(n, 0, range * eps)
    x = x + rnorm(length(x), 0, diff(range(x)) * eps)
  } else if (is.factor(x)) {
    # For factors we randomly flip eps %
    flip = sample(c(TRUE, FALSE), length(x), replace = TRUE, prob = c(eps, 1 - eps))
    x[flip] = sample(x, flip)
  }
  return(x)
}

