library(mlr)
devtools::load_all()
lrn = makeLearner("classif.xgboost.custom", predict.type = "prob", nrounds = 10L)
mod = train(learner = lrn, task = iris.task)
task = iris.task
measures = list(acc, timetrain)
n_classes = length(task$task.desc$class.levels)

mod$learner$par.vals$ntreelimit = 1
performance(predict(mod, task), model = mod, measures = measures)


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


# Earlier Hyperpars
early_stopping_rounds = function(value) {
  if (missing(value)) {
    return(self$pipeline$early_stopping_rounds)
  } else {
    self$pipeline$early_stopping_rounds = assert_integerish(value, lower = 1L, len = 1L)
    return(self)
  }
},
early_stopping_fraction = function(value) {
  if (missing(value)) {
    return(self$pipeline$early_stopping_fraction)
  } else {
    self$pipeline$early_stopping_fraction = assert_numeric(value, lower = 0, upper = 1, len = 1L)
    return(self)
  }
},
    tune_threshold = function(value) {
  if (missing(value)) {
    return(self$pipeline$tune_threshold)
  } else {
    self$pipeline$tune_threshold = assert_flag(value)
    return(self)
  }
},

   max_nrounds = function(value) {
      if (missing(value)) {
        return(self$pipeline$max_nrounds)
      } else {
        self$pipeline$max_nrounds = assert_integerish(value, lower = 1L, len = 1L)
      }
    },
    impact_encoding_boundary = function(value) {
      if (missing(value)) {
        return(self$pipeline$impact_encoding_boundary)
      } else {
        self$pipeline$impact_encoding_boundary = assert_integerish(value, lower = 1L, len = 1L)
      }
    },
    nthread = function(value) {
      if (missing(value)) {
        return(self$pipeline$nthread)
      } else {
        self$pipeline$nthread = assert_integerish(value, lower = 1, len = 1L, null.ok = TRUE)
        return(self)
      }
    },
    resample_instance = function(value) {
      if (missing(value)) {
        return(self$pipeline$resample_instance)
      } else {
        self$pipeline$resample_instance = assert_class(value, "ResampleInstance", null.ok = TRUE)
        return(self)
      }
    },
#' @param early_stopping_measure [\code{\link[mlr]{Measure}}]\cr
#'   Performance measure used for early stopping. Picks the first measure
#'   defined in measures by default.
#' @param early_stopping_rounds [\code{integer(1L}]\cr
#'   After how many iterations without an improvement in the boosting OOB error should be stopped?
#'   Default is \code{10}.
#' @param early_stopping_fraction [\code{numeric(1)}]\cr
#'   What fraction of the data should be used for early stopping (i.e. as a validation set).
#'   Default is \code{4/5}.
#' Additional arguments that control the Pipeline:
#' @param impact_encoding_boundary [\code{integer(1)}]\cr
#'   Defines the threshold on how factor variables are handled. Factors with more levels than the \code{"impact_encoding_boundary"} get impact encoded while factor variables with less or equal levels than the \code{"impact_encoding_boundary"} get dummy encoded.
#'   For \code{impact_encoding_boundary = 0L}, all factor variables get impact encoded while for \code{impact_encoding_boundary = .Machine$integer.max}, all of them get dummy encoded.
#'   Default is \code{10}.
#' @param tune_threshold [logical(1)]\cr
#'   Should thresholds be tuned? This has only an effect for classification, see \code{\link[mlr]{tuneThreshold}}.
#'   Default is \code{TRUE}.
#' @param max_nrounds [\code{integer(1)}]\cr
#'   Maximum number of allowed boosting iterations. Default is \code{3000}.
#'
