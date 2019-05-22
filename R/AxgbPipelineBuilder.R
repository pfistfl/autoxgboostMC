#' @title Abstract Base Class
#' @seealso \code{\link{AxgbPipelineBuilderXGB}}
#' @export
AxgbPipelineBuilder = R6::R6Class("AxgbPipelineBuilder",
  public = list(
    initialize = function() {stop("Abstract Base class!")},
    configure = function(logger) {
      private$.logger  = assert_class(logger, "logger")
    },
    build_transform_pipeline = function{stop("Abstract Base class!")},
    make_baselearner = function(){stop("Abstract Base class!")},
  ),
  private = list(
    .logger = NULL
  )
)

#' @title Build a xgboost baselearner and Preproc Pipeline
#'
#' Additional arguments that control the Pipeline can be set via Active Bindings.
#'
#' @param early_stopping_measure [\code{\link[mlr]{Measure}}]\cr
#'   Performance measure used for early stopping. Picks the first measure
#'   defined in measures by default.
#' @param early_stopping_rounds [\code{integer(1L}]\cr
#'   After how many iterations without an improvement in the boosting OOB error should be stopped?
#'   Default is \code{10}.
#' @param early_stopping_fraction [\code{numeric(1)}]\cr
#'   What fraction of the data should be used for early stopping (i.e. as a validation set).
#'   Default is \code{4/5}.
#' @param impact_encoding_boundary [\code{integer(1)}]\cr
#'   Defines the threshold on how factor variables are handled. Factors with more levels than the \code{"impact_encoding_boundary"} get impact encoded while factor variables with less or equal levels than the \code{"impact_encoding_boundary"} get dummy encoded.
#'   For \code{impact_encoding_boundary = 0L}, all factor variables get impact encoded while for \code{impact_encoding_boundary = .Machine$integer.max}, all of them get dummy encoded.
#'   Default is \code{10}.
#' @param tune_threshold [logical(1)]\cr
#'   Should thresholds be tuned? This has only an effect for classification, see \code{\link[mlr]{tuneThreshold}}.
#'   Default is \code{TRUE}.
#' @param max_nrounds [\code{integer(1)}]\cr
#'   Maximum number of allowed boosting iterations. Default is \code{3000}.
#' @export
AxgbPipelineBuilderXGB = R6::R6Class("AxgbPipelineBuilderXGB",
  inherit = AxgbPipelineBuilder,
  public = list(
  task_type = NULL,
  baselearner = NULL,
  preproc_pipeline = NULL,

  initialize = function(logger) {
    private$.logger  = assert_class(logger, "logger")
  },
  make_baselearner = function(task, measures, nthread, maximize_es_measure) {
    self$make_baselearner_earlystop(task, measures, nthread, maximize_es_measure)
  },
  make_baselearner_earlystop = function(task, measures, nthread, maximize_es_measure) {
      private$.nthread = assert_integerish(nthread, lower = 1, len = 1L, null.ok = TRUE)
      self$task_type = getTaskType(task)
      td = getTaskDesc(task)
      req_prob_measure = sapply(measures, function(x) {
        any(getMeasureProperties(x) == "req.prob")
      })

      pv = list()
      if (!is.null(private$.nthread)) pv$nthread = private$.nthread

      if (self$task_type == "classif") {
        predict.type = ifelse(any(req_prob_measure) | private$.tune_threshold, "prob", "response")
        if(length(td$class.levels) == 2) {
          objective = "binary:logistic"
          eval_metric = "error"
          parset = c(self$parset, makeParamSet(makeNumericParam("scale_pos_weight", lower = -10, upper = 10, trafo = function(x) 2^x)))
        } else {
          objective = "multi:softprob"
          eval_metric = "merror"
        }
        baselearner = makeLearner("classif.xgboost.earlystop", id = "classif.xgboost.earlystop",
          predict.type = predict.type, eval_metric = eval_metric, objective = objective,
          early_stopping_rounds = private$.early_stopping_rounds, maximize = maximize_es_measure,
          max.nrounds = private$.max_nrounds, par.vals = pv)

      } else if (self$task_type == "regr") {
        predict.type = NULL
        objective = "reg:linear"
        eval_metric = "rmse"
        baselearner = makeLearner("regr.xgboost.earlystop", id = "regr.xgboost.earlystop",
          eval_metric = eval_metric, objective = objective, early_stopping_rounds = private$.early_stopping_rounds,
          maximize = maximize_es_measure, max.nrounds = private$.max_nrounds, par.vals = pv)
      } else {
        stop("Task must be regression or classification")
      }
      self$baselearner = setHyperPars(baselearner, early.stopping.data = private$.early_stopping_data)
      return(self$baselearner)
    },
    build_transform_pipeline = function(task) {
      self$build_pipeline(task)
      tfed_tasks = self$transform_with_pipeline(task)
      return(tfed_tasks)
    },
    # Build pipeline
    build_pipeline = function(task) {
      has_cat_feats = sum(getTaskDesc(task)$n.feat[c("factors", "ordered")]) > 0
      preproc_pipeline = NULLCPO
      if (has_cat_feats) {
        preproc_pipeline %<>>% generateCatFeatPipeline(task, private$.impact_encoding_boundary)
      }
      preproc_pipeline %<>>% cpoDropConstants()

      # Store built pipeline.
      self$preproc_pipeline = preproc_pipeline
    },
    transform_with_pipeline = function(task) {
      # The pipeline stays constant during training. As a result, we preprocess data here
      # once and split early stopping data.
      if (is.null(private$.resample_instance))
        private$.resample_instance = makeResampleInstance(makeResampleDesc("Holdout", split = private$.early_stopping_fraction), task)
      train_task = subsetTask(task, private$.resample_instance$train.inds[[1]])
      test_task =  subsetTask(task, private$.resample_instance$test.inds[[1]])
      train_task %<>>% self$preproc_pipeline
      test_task %<>>% retrafo(train_task)
      private$.early_stopping_data = test_task
      return(list(train_task = train_task, test_task = test_task))
    },
    build_final_learner = function(pars) {
      if (self$task_type == "classif") {
        lrn = makeLearner("classif.xgboost.custom", nrounds = pars$nrounds,
          objective = self$baselearner$par.vals$objective,
          predict.type = self$baselearner$predict.type,
          predict.threshold = pars$threshold)
      } else {
        lrn = makeLearner("regr.xgboost.custom", nrounds = nrounds, objective = self$baselearner$par.vals$objective)
      }
      lrn = setHyperPars2(lrn, par.vals = pars)
      lrn = self$preproc_pipeline %>>% lrn
      return(lrn)
    }
  ),
  private = list(
    .max_nrounds = 3*10^3L,
    .early_stopping_rounds = 20L,
    .early_stopping_data = NULL,
    .early_stopping_fraction = 4/5,
    .impact_encoding_boundary = 10L,
    .resample_instance = NULL,
    .tune_threshold = TRUE,
    .nthread = NULL
  ),
  active = list(
    resample_instance = function() {private$.resample_instance},
    max_nrounds = function(value) {
      if (missing(value)) {
        return(private$.max_nrounds)
      } else {
        private$.max_nrounds = assert_integerish(value, lower = 1L, len = 1L)
      }
    },
    early_stopping_rounds = function(value) {
      if (missing(value)) {
        return(private$.early_stopping_rounds)
      } else {
        private$.early_stopping_rounds = assert_integerish(value, lower = 1L, len = 1L)
      }
    },
    early_stopping_fraction = function(value) {
      if (missing(value)) {
        return(private$.early_stopping_fraction)
      } else {
        private$.early_stopping_fraction = assert_numeric(value, lower = 0, upper = 1, len = 1L)
      }
    },
    impact_encoding_boundary = function(value) {
      if (missing(value)) {
        return(private$.impact_encoding_boundary)
      } else {
        private$.impact_encoding_boundary = assert_numeric(value, lower = 0, upper = 1, len = 1L)
      }
    },
    tune_threshold = function(value) {
      if (missing(value)) {
        return(private$.tune_threshold)
      } else {
        private$.tune_threshold = assert_flag(value)
      }
    },
    nthread = function(value) {
      if (missing(value)) {
        return(private$.nthread)
      } else {
        private$.nthread = assert_integerish(value, lower = 1, len = 1L, null.ok = TRUE)
      }
    },
    logger = function(value) {
      if (missing(value)) {
        return(private$.logger)
      } else {
        private$.logger = assert_integerish(value, lower = 1, len = 1L, null.ok = TRUE)
      }
    }
  )
)
