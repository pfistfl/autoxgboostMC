#' @title Abstract Base Class
#' @seealso \code{\link{AxgbPipelineBuilderXGB}}
#' @export
AxgbPipelineBuilder = R6::R6Class("AxgbPipelineBuilder",
  public = list(
    initialize = function() {stop("Abstract Base class!")},
    configure = function(logger, parset) {
      if (is.null(private$.logger))
        private$.logger = assert_class(logger, "logger")
      if (is.null(private$.parset))
        private$.parset =  coalesce(parset, autoxgboostMC::autoxgbparset)
    },
    build_transform_pipeline = function() {stop("Abstract Base class!")},
    make_baselearner =         function() {stop("Abstract Base class!")},
    make_objective_function =  function() {stop("Abstract Base class!")}
  ),
  private = list(
    .logger = NULL,
    .parset = NULL
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

  initialize = function() {
  },
  get_objfun = function(task, measures, parset, nthread) {
    assert_class(task, "SupervisedTask")
    assert_class(parset, "ParamSet", null.ok = TRUE)
    private$.measures = assert_list(measures, types = "Measure", null.ok = TRUE)
    private$.parset = coalesce(parset, autoxgboostMC::autoxgbparset)
    private$.parset = c(private$.parset, private$make_subeval_parset(task))

    self$preproc_pipeline = self$build_pipeline(task)
    transf_tasks = self$transform_with_pipeline(task)
    private$.baselearner = self$make_baselearner(task, nthread)
    obj_fun = self$make_objective_function_multicrit(transf_tasks)
    return(obj_fun)
  },
  make_baselearner = function(task, nthread) {
      private$.nthread = assert_integerish(nthread, lower = 1, len = 1L, null.ok = TRUE)
      self$task_type = getTaskType(task)
      td = getTaskDesc(task)

      pv = list()
      if (!is.null(private$.nthread)) pv$nthread = private$.nthread

      if (self$task_type == "classif") {
        if(length(td$class.levels) == 2) {
          objective = "binary:logistic"
          eval_metric = "error"
          private$.parset = c(private$.parset, makeParamSet(makeNumericParam("scale_pos_weight", lower = -10, upper = 10, trafo = function(x) 2^x)))
        } else {
          objective = "multi:softprob"
          eval_metric = "merror"
        }
        baselearner = makeLearner("classif.xgboost.custom", id = "classif.xgboost.custom",
          predict.type = "prob", eval_metric = eval_metric, objective = objective, par.vals = pv)
      } else if (self$task_type == "regr") {
        predict.type = NULL
        objective = "reg:linear"
        eval_metric = "rmse"
        baselearner = makeLearner("regr.xgboost.custom", id = "regr.xgboost.custom",
          eval_metric = eval_metric, objective = objective, par.vals = pv)
      } else {
        stop("Task must be regression or classification")
      }
      return(baselearner)
    },
    # Build pipeline
    build_pipeline = function(task) {
      has_cat_feats = sum(getTaskDesc(task)$n.feat[c("factors", "ordered")]) > 0
      preproc_pipeline = NULLCPO
      if (has_cat_feats) {
        preproc_pipeline %<>>% generateCatFeatPipeline(task, private$.impact_encoding_boundary)
      }
      preproc_pipeline %<>>% cpoDropConstants()
      return(preproc_pipeline)
    },
    transform_with_pipeline = function(task) {
      # The pipeline stays constant during training. As a result, we preprocess data here
      # once and split early stopping data.
      if (is.null(private$.resample_instance))
        private$.resample_instance = makeResampleInstance(makeResampleDesc("Holdout", split = 0.75), task)
      train_task = subsetTask(task, private$.resample_instance$train.inds[[1]])
      test_task  = subsetTask(task, private$.resample_instance$test.inds[[1]])
      train_task %<>>% self$preproc_pipeline
      test_task  %<>>% retrafo(train_task)
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
    },
    make_objective_function_multicrit = function(transf_tasks) {
      smoof::makeMultiObjectiveFunction(name = "optimizeWrapperMultiCrit",
      fn = function(x, subevals = FALSE) {
        x = x[!vlapply(x, is.na)]
        lrn = setHyperPars(private$.baselearner, par.vals = x[vlapply(names(x), `!=`, "threshold")])
        mod = train(lrn, transf_tasks$train_task)
        pred = predict(mod, transf_tasks$test_task)
        pred = setThreshold(pred, x$threshold)
        res = performance(pred, model = mod, task = transf_tasks$test_task, measures = private$.measures)

        if (subevals) attr(res, "extras") = list(.subevals = private$get_subevals(mod, transf_tasks$test_task, private$.measures))
        return(res)
      },
      par.set = private$.parset, n.objectives = length(private$.measures), minimize = self$measures_minimize,
      noisy = FALSE, has.simple.signature = FALSE)
    }
  ),
  private = list(
    get_subevals = function(mod, task, measures) {
      n_classes = length(task$task.desc$class.levels)

      # Compute a set of different thresholds and nrounds
      ncomb = ceiling(100^(1 / n_classes))
      threshold_vals = mlrMBO:::combWithSum(ncomb, n_classes) / ncomb
      if (n_classes > 2L) {
        threshold_vals = rbind(threshold_vals, 1 / n_classes)
        colnames(threshold_vals) = task$task.desc$class.levels
      } else {
        threshold_vals = threshold_vals[, 1, drop = FALSE]
      }

      thresholds = BBmisc::convertRowsToList(threshold_vals, name.vector = TRUE)
      ntreelimit_vals  = quantile(seq_len(mod$learner$par.vals$nrounds), probs = c(25, 50, 75, 100)/100, type = 1)

      # Compute performances
      lst = lapply(ntreelimit_vals, function(ntreelimit) {
        # Predict with ntreelimit < nrounds trees
        mod$learner$par.vals$ntreelimit = ntreelimit
        prd = predict(mod, task = task)
        # Predict for different thresholds
        do.call("rbind", lapply(thresholds, function(threshold) {
          prd = setThreshold(prd, threshold)
          performance(prd, model = mod, measures = measures)
        }))
      })
      list(
        y = do.call("rbind", lst),
        x = data.frame(do.call("rbind", lapply(ntreelimit_vals, function(x) cbind(ntreelimit = x, threshold_vals))))
      )
    },
    make_subeval_parset = function(task) {
      threshold_len = length(task$task.desc$class.levels)
      if (threshold_len == 2L) threshold_len = 1L
      makeParamSet(
        makeNumericVectorParam("threshold", lower = 0, upper = 1, len = threshold_len, trafo = function(x) {
         if(length(x) > 1L) x = x / sum(x)
         return(x)
        }),
        makeIntegerParam(id = "nrounds", lower = 1L, upper = private$.max_nrounds)
      )
    },
    .max_nrounds = 300L,
    .impact_encoding_boundary = 10L,
    .resample_instance = NULL,
    .nthread = NULL,
    .measures = NULL,
    .baselearner = NULL
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
    },
    measures_minimize = function() {sapply(private$.measures, function(x) x$minimize)}
  )
)

