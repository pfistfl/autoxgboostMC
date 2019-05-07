### Fairness Measures:
# I) Independence
#' Absolute differences of Positive Rate between groups
#'
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

#' Absolute differences of F1 Scores between groups
#' See Hardt et al, 2016: https://arxiv.org/pdf/1610.02413.pdf
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




## 3) Calibration
#' Absolute differences of Positive Predictive Value between groups
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
# FIXME: We have to compute 'imeasure' globally only once. Or aggregate, otherwise this is very slow.

#' Curve Complexity
#' For more info see Molnar et al., 2019 https://arxiv.org/abs/1904.03867
#' @export
interpmec = mlr::makeMeasure(id = "interp.mec", minimize = TRUE, properties = c("classif", "response", "req.task"),
  extra.args = list(), best = 0, worst = 1,
  fun = function(task, model, pred, feats, extra.args) {
    task_data = getTaskData(task)
    pred = iml::Predictor$new(model, task_data, y = task$task.desc$target)
    pred$class = 1L
    imeasure = autoiml:::FunComplexity$new(pred, max_seg_cat = 0.05, max_seg_num = 5, epsilon = 0.05, grid.size = 10)
    return(round(imeasure$c_wmean, 1))
  }
)

#' Interaction Strength
#' For more info see Molnar et al., 2019 https://arxiv.org/abs/1904.03867
#' @export
interpias = mlr::makeMeasure(id = "interp.ias", minimize = TRUE, properties = c("classif", "response", "req.task"),
  extra.args = list(), best = 0, worst = 1,
  fun = function(task, model, pred, feats, extra.args) {
    task_data = getTaskData(task)
    pred = iml::Predictor$new(model, task_data, y = task$task.desc$target)
    pred$class = 1L
    imeasure = autoiml:::FunComplexity$new(pred, max_seg_cat = 0.05, max_seg_num = 5, epsilon = 0.05, grid.size = 10)
    return(imeasure$n_features)
  }
)

#' Number of features
#' For more info see Molnar et al., 2019 https://arxiv.org/abs/1904.03867
#' @export
interpnf = mlr::makeMeasure(id = "interp.nfeat", minimize = TRUE, properties = c("classif", "response", "req.task"),
  extra.args = list(), best = 0, worst = 1,
  fun = function(task, model, pred, feats, extra.args) {
    task_data = getTaskData(task)
    pred = iml::Predictor$new(model, task_data, y = task$task.desc$target)
    pred$class = 1L
    imeasure = autoiml:::FunComplexity$new(pred, max_seg_cat = 0.05, max_seg_num = 5, epsilon = 0.05, grid.size = 10)
    return(imeasure$n_features)
  }
)
### Robustness Measures:

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
robustnoise = mlr::makeMeasure(id = "robustness.noise", minimize = FALSE, properties = c("classif", "response", "req.task"),
  extra.args = list(eps = 0.01, n = 1L), best = 0, worst = 1,
  fun = function(task, model, pred, feats, extra.args) {
    eps = assert_numeric(extra.args$eps)
    reps = assert_integerish(n)
    # Repeat n times for robustness.
    res = vnapply(seq_len(n), function() {
      noise_pred = predict(model, newdata = inject_noise(task, eps))
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
robustnoiseperfeat = mlr::makeMeasure(id = "robustness.perfeat.noise", minimize = FALSE, properties = c("classif", "response", "req.task"),
  extra.args = list(eps = 0.01, n = 1L), best = 0, worst = 1,
  fun = function(task, model, pred, feats, extra.args) {
    eps = assert_numeric(extra.args$eps)
    reps = assert_integerish(n)
    # Repeat n times for robustness.
    res = vnapply(seq_len(n), function() {
      perfeat_res = vnapply(seq_len(getTaskNFeats(task)), function(feature) {
        noise_pred = predict(model, newdata = inject_noise_single_feature(task, eps))
        mean(pred$data$response == noise_pred$data$response, na.rm = TRUE)
      })
      mean(perfeat_res)
    })
    mean(res)
  }
)

# Adversarial Examples


### Helper functions

#' Obtain a grouping factor from either a data column, an additional factor or a function.
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

#' Inject noise into every feature of a dataset.
inject_noise_task = function(task, eps = 0.01) {
  feats = getTaskData(task, target.extra = TRUE)$data
  feats = sapply(feats, inject_noise, eps = eps)
  return(feats)
}

#' Inject noise into a single feature of a dataset.
inject_noise_single_feature = function(task, feature, eps = 0.05) {
  assert_integerish(feature)
  feats = getTaskData(task, target.extra = TRUE)$data
  feats[, feature, drop = FALSE] = inject_noise(feats[, feature, drop = FALSE])
  return(feats)
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
}

