#' @title Fit and optimize a xgboost model for multiple criteria
#'
#' @description
#' An xgboost model is optimized based on a set of measures (see [\code{\link[mlr]{Measure}}]).
#' The bounds of the parameter in which the model is optimized, are defined by \code{\link{autoxgbparset}}.
#' For the optimization itself Bayesian Optimization with \pkg{mlrMBO} is used.
#' Without any specification of the control object, the optimizer runs for for 160 iterations or 1 hour,
#' whichever happens first.
#' Both the parameter set and the control object can be set by the user.
#'
#' Arguments to `.$new()`:
#' @param task [\code{\link[mlr]{Task}}]\cr
#'   The task to be trained.
#' @param measures [list of \code{\link[mlr]{Measure}}]\cr
#'   Performance measure. If \code{NULL} \code{\link[mlr]{getDefaultMeasure}} is used.
#' @param parset [\code{\link[ParamHelpers]{ParamSet}}]\cr
#'   Parameter set to tune over. Default is \code{\link{autoxgbparset}}.
#'   Can be updated using `.$set_parset()`.
#' @param nthread [integer(1)]\cr
#'   Number of cores to use.
#'   If \code{NULL} (default), xgboost will determine internally how many cores to use.
#'   Can be set using `.$set_nthread()`.
#'
#' Arguments to `.$fit()`:
#' @param iterations [\code{integer(1L}]\cr
#'   Number of MBO iterations to do. Will be ignored if a custom \code{MBOControl} is used.
#'   Default is \code{160}.
#' @param time_budget [\code{integer(1L}]\cr
#'   Time that can be used for tuning (in seconds). Will be ignored if a custom \code{control} is used.
#'   Default is \code{3600}, i.e., one hour.
#' @param fit_final_model [\code{logical(1)}]\cr
#'   Should the model with the best found configuration be refitted on the complete dataset?
#'   Default is \code{FALSE}. The model can also be fitted after optimization using `.$fit_final_model()`.
#' @param plot [\code{logical(1)}]\cr
#'   Should the progress be plotted? Default is \code{TRUE}.
#'
#' Additional arguments that control the Bayesian Optimization process:
#' Can be set / obtained via respective Active Bindings:
#' @param control [\code{\link[mlrMBO]{MBOControl}}]\cr
#'   Control object for optimizer.
#'   If not specified, the default \code{\link[mlrMBO]{makeMBOControl}}] object will be used with
#'   \code{iterations} maximum iterations and a maximum runtime of \code{time_budget} seconds.
#' @param mbo_learner [\code{\link[mlr]{Learner}}]\cr
#'   Regression learner from mlr, which is used as a surrogate to model our fitness function.
#'   If \code{NULL} (default), the default learner is determined as described here:
#'   \link[mlrMBO]{mbo_default_learner}.
#' @param design_size [\code{integer(1)}]\cr
#'   Size of the initial design. Default is \code{15L}.
#'
#' Additional arguments that control the Pipeline:
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
#'
#' @export
#' @examples
#' \donttest{
#' # Create a mlr Task
#' iris.task = makeClassifTask(data = iris, target = "Species")
#' # Instantiate the AutoxgboostMC Object
#' axgb = AutoxgboostMC$new(iris.task, measure = auc)
#' # Fit and Predict
#' axgb$fit(time_budget = 5L)
#' p = axgb$predict(iris.task)
#'
#' # Set hyperparameters:
#' axgb$tune_threshold = FALSE
#' }
AutoxgboostMC = R6::R6Class("AutoxgboostMC",
  public = list(
    task = NULL,
    measures = NULL,

    iterations = NULL,
    time_budget = NULL,

    preproc_pipeline = NULL,
    obj_fun = NULL,
    opt_state = NULL,
    opt_result = NULL,
    optimizer_constructor = AxgbOptimizerSMBO,
    optimizer = NULL,

    final_learner = NULL,
    final_model = NULL,

    initialize = function(task, measures = NULL, parset = NULL, nthread = NULL) {
      self$task = assert_class(task, "SupervisedTask")
      assert_list(measures, types = "Measure", null.ok = TRUE)
      assert_class(parset, "ParamSet", null.ok = TRUE)
      # Set defaults
      measures = coalesce(measures, list(getDefaultMeasure(task)))
      # names(measures) = self$measure_ids
      self$measures = lapply(measures, self$set_measure_bounds)
      private$.parset = coalesce(parset, autoxgboostMC::autoxgbparset)
      private$.nthread = assert_integerish(nthread, lower = 1, len = 1L, null.ok = TRUE)
      private$.logger = log4r::logger(threshold = "WARN")
      private$.watch = Stopwatch$new(time_budget, iterations)

      private$baselearner = self$make_baselearner()
      transf_tasks = self$build_transform_pipeline()
      private$baselearner = setHyperPars(private$baselearner, early.stopping.data = transf_tasks$task.test)
      self$obj_fun = self$make_objective_function(transf_tasks)

      self$optimizer = self$optimizer_constructor$new(self$obj_fun, self$parset, private$.logger, private$.watch)
    },
    print = function(...) {
      catf("AutoxgboostMC Learner")
      catf("Task: %s (%s)", self$task$task.desc$id, self$task$type)
      catf("Measures: %s", paste0(self$measure_ids, collapse = ","))
      catf("Trained: %s", ifelse(is.null(self$opt_result), "no", "yes"))
      if (!is.null(self$opt_result)) {
        op = self$opt_result$opt.path
        pars = trafoValue(op$par.set, self$opt_result$x)
        pars$nrounds = self$get_best_from_opt("nrounds")
        catf("Autoxgboost tuning result")
        catf("Recommended parameters:")
        for (p in names(pars)) {
          if (p == "nrounds" || isInteger(op$par.set$pars[[p]])) {
            catf("%s: %i", stringi::stri_pad_left(p, width = 17), as.integer(pars[p]))
          } else if (isNumeric(op$par.set$pars[[p]], include.int = FALSE)) {
            catf("%s: %.3f", stringi::stri_pad_left(p, width = 17), pars[p])
          } else {
            catf("%s: %s", stringi::stri_pad_left(p, width = 17), pars[p])
          }
        }
        catf("\n\nPreprocessing pipeline:")
            print(self$preproc_pipeline)
        # FIXME: Nice Printer for results:
        catf("\nWith tuning result:")
        for (i in seq_along(self$measures)) catf("    %s = %.3f", self$measures[[i]]$id, self$opt_result$y[[i]])
        thr = self$get_best_from_opt(".threshold")
        if (!is.null(thr)) {
          if (length(thr) == 1) {
            catf("\nClassification Threshold: %.3f", thr)
          } else {
            catf("\nClassification Thresholds: %s", paste(names(thr), round(thr, 3), sep = ": ", collapse = "; "))
          }
        }
      }
    },
    fit = function(iterations = 160L, time_budget = 3600L, fit_final_model = TRUE, plot = TRUE) {
      assert_integerish(iterations)
      assert_integerish(time_budget)
      assert_flag(fit_final_model)
      assert_flag(plot)

      # Initialize  the optimizer
      self$optimizer$fit(iterations, time_budget, plot)

      # Create final model
      self$final_learner = self$build_final_learner()
      if(fit_final_model) self$fit_final_model()
    },
    predict = function(newdata) {
      assert(check_class(newdata, "data.frame"), check_class(newdata, "Task"), combine = "or")
      if(is.null(self$final_model)) stop("Final model not fitted, use .$fit_final_model() to fit!")
      predict(self$final_model, newdata)
    },
    fit_final_model = function() {
      if(is.null(self$final_learner)) self$build_final_learner()
      self$final_model = train(self$final_learner, self$task)
    },

    # AutoxgboostMC steps
    make_baselearner = function() {
      tt = getTaskType(self$task)
      td = getTaskDesc(self$task)
      req_prob_measure = sapply(self$measures, function(x) {
        any(getMeasureProperties(x) == "req.prob")
      })

      pv = list()
      if (!is.null(self$nthread)) pv$nthread = self$nthread

      if (tt == "classif") {
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
          early_stopping_rounds = private$.early_stopping_rounds, maximize = !self$early_stopping_measure$minimize,
          max.nrounds = private$.max_nrounds, par.vals = pv)

      } else if (tt == "regr") {
        predict.type = NULL
        objective = "reg:linear"
        eval_metric = "rmse"
        baselearner = makeLearner("regr.xgboost.earlystop", id = "regr.xgboost.earlystop",
          eval_metric = eval_metric, objective = objective, early_stopping_rounds = private$.early_stopping_rounds,
          maximize = !self$early_stopping_measure$minimize, max.nrounds = private$.max_nrounds, par.vals = pv)
      } else {
        stop("Task must be regression or classification")
      }
      return(baselearner)
    },

    # Build pipeline
    build_transform_pipeline = function() {
      has.cat.feats = sum(getTaskDesc(self$task)$n.feat[c("factors", "ordered")]) > 0
      self$preproc_pipeline = NULLCPO
      if (has.cat.feats) {
        self$preproc_pipeline %<>>% generateCatFeatPipeline(self$task, private$.impact_encoding_boundary)
      }
      self$preproc_pipeline %<>>% cpoDropConstants()

      # process data and apply pipeline
      # split early stopping data
      if (is.null(private$.resample_instance))
        private$.resample_instance = makeResampleInstance(makeResampleDesc("Holdout", split = private$.early_stopping_fraction), self$task)

      task.test =  subsetTask(self$task, private$.resample_instance$test.inds[[1]])
      task.train = subsetTask(self$task, private$.resample_instance$train.inds[[1]])

      task.train %<>>% self$preproc_pipeline
      task.test %<>>% retrafo(task.train)
      return(list(task.train = task.train, task.test = task.test))
    },

    # MBO --------------------------------------------------------------------------------
    make_objective_function = function(transf_tasks) {
      is_thresholded_measure = sapply(self$measures, function(x) {
        props = getMeasureProperties(x)
        any(props == "req.truth") & !any(props == "req.prob")
      })
      if (!any(is_thresholded_measure) & private$.tune_threshold) {
        warning("Threshold tuning is active, but no measure for tuning thresholds!
          Deactivating threshold tuning!")
        private$.tune_threshold = FALSE
      }

      smoof::makeMultiObjectiveFunction(name = "optimizeWrapperMultiCrit",
        fn = function(x) {
          x = x[!vlapply(x, is.na)]
          lrn = setHyperPars(private$baselearner, par.vals = x)
          mod = train(lrn, transf_tasks$task.train)
          pred = predict(mod, transf_tasks$task.test)
          nrounds = self$get_best_iteration(mod)
          # For now we tune threshold of first applicable measure.
          if (private$.tune_threshold && getTaskType(transf_tasks$task.train) == "classif") {
            tune.res = tuneThreshold(pred = pred, measure = self$measures[is_thresholded_measure][[1]])

            if (length(self$measures[-which(is_thresholded_measure)[1]]) > 0) {
              res = performance(pred, self$measures[-which(is_thresholded_measure)[1]], model = mod, task = transf_tasks$task.test)
              res = c(res, tune.res$perf)
            } else {
              res = tune.res$perf
            }
            attr(res, "extras") = list(nrounds = nrounds, .threshold = tune.res$th)
          } else {
            res = performance(pred, self$measures, model = mod, task = transf_tasks$task.test)
            attr(res, "extras") = list(nrounds = nrounds)
          }

          return(res)
        },
        par.set = private$.parset, noisy = FALSE, has.simple.signature = FALSE, minimize = self$measure_minimize,
        n.objectives = length(self$measures)
      )
    },
    build_final_learner = function() {
      nrounds = self$get_best_from_opt("nrounds")
      pars = trafoValue(self$parset, self$opt_result$x)
      pars = pars[!vlapply(pars, is.na)]

      if (!is.null(private$baselearner$predict.type)) {
        lrn = makeLearner("classif.xgboost.custom", nrounds = nrounds,
          objective = private$baselearner$par.vals$objective,
          predict.type = private$baselearner$predict.type,
          predict.threshold = self$get_best_from_opt(".threshold"))
      } else {
        lrn = makeLearner("regr.xgboost.custom", nrounds = nrounds, objective = objective)
      }
      lrn = setHyperPars2(lrn, par.vals = pars)
      lrn = self$preproc_pipeline %>>% lrn
      return(lrn)
    },

    ## Setters for various hyperparameters -----------------------------------------------
    set_measure_bounds = function(measure, best_valid = NULL, worst_valid = NULL) {
      if(is.null(best_valid)  & is.null(measure$best_valid))  measure$best_valid = measure$best
      else measure$best_valid = best_valid
      if(is.null(worst_valid) & is.null(measure$worst_valid)) measure$best_valid = measure$best
      else measure$worst_valid = worst_valid
      if(is.null(measure$weight)) measure$weight = 1L
      return(measure)
    },
    set_hyperpars = function(par_vals) {
      assert_list(par_vals, names = "unique")
      lapply(names(par_vals), function(x) self[[x]] = par_vals[[x]])
      invisible(self)
    },
    ## Getters
    # Get best value from optimization result
    # @param what [`character(1)`]: "nrounds" or ".threshold"
    get_best_from_opt = function(what) {
      self$opt_result$opt.path$env$extra[[self$get_best_ind(self$opt_result)]][[what]]
    },
    # Get the iteration parameter of a fitted xboost model with early stopping
    get_best_iteration = function(mod) {
      getLearnerModel(mod, more.unwrap = TRUE)$best_iteration
    },
    get_best_ind = function(opt_result) {
      if (self$early_stopping_measure$minimize) {
        best.ind = which.min(opt_result$opt.path$env$path[[self$early_stopping_measure$id]])
      } else {
        best.ind = which.max(opt_result$opt.path$env$path[[self$early_stopping_measure$id]])
      }
      return(best.ind)
    },
    get_opt_path_df = function() {
      as.data.frame(mlrMBO:::getOptStateOptPath(self$opt_state))
    },
  ),

  active = list(
    early_stopping_measure = function(value) {
      if (missing(value)) {
        self$measures[[1]]
      } else {
        measure_ids = sapply(self$measures, function(x)  x$id)
        assert_list(value, types = "Measure", null.ok = TRUE)
        self$measures = c(value, self$measures[-which(value$id == measure_ids)])
        messagef("Setting %s as early stopping measure!", value$id)
      }
    },
    is_multicrit = function() {
      length(self$measures) > 1
    },
    measure_ids = function() {
      sapply(self$measures, function(x) x$id)
    },
    measure_minimize = function() {
      sapply(self$measures, function(x) x$minimize)
    },

    # Hyperparameters --------------------------------------------------------------------
    max_nrounds = function(value) {
      if (missing(value)) {
        return(private$.max_nrounds)
      } else {
        private$.max_nrounds = assert_integerish(value, lower = 1L, len = 1L)
        return(self)
      }
    },
    early_stopping_rounds = function(value) {
      if (missing(value)) {
        return(private$.early_stopping_rounds)
      } else {
        private$.early_stopping_rounds = assert_integerish(value, lower = 1L, len = 1L)
        return(self)
      }
    },
    early_stopping_fraction = function(value) {
      if (missing(value)) {
        return(private$.early_stopping_fraction)
      } else {
        private$.early_stopping_fraction = assert_numeric(value, lower = 0, upper = 1, len = 1L)
        return(self)
      }
    },
    impact_encoding_boundary = function(value) {
      if (missing(value)) {
        return(private$.impact_encoding_boundary)
      } else {
        private$.impact_encoding_boundary = assert_numeric(value, lower = 0, upper = 1, len = 1L)
        return(self)
      }
    },
    tune_threshold = function(value) {
      if (missing(value)) {
        return(private$.tune_threshold)
      } else {
        private$.tune_threshold = assert_flag(value)
        return(self)
      }
    },
    nthread = function(value) {
      if (missing(value)) {
        return(private$.nthread)
      } else {
        self$nthread = assert_integerish(value, lower = 1, len = 1L, null.ok = TRUE)
        return(self)
      }
    },
    resample_instance = function(value) {
      if (missing(value)) {
        return(private$.resample_instance)
      } else {
        private$.resample_instance = assert_class(value, "ResampleInstance", null.ok = TRUE)
        return(self)
      }
    },
    logger = function(value) {
        if (missing(value)) {
        return(private$.logger)
      } else {
        private$.logger = assert_class(value, "logger")
        return(self)
      }
    },
    watch = function() {private$.watch},
    # MBO Hyperparameters
    parset = function(value) {
      if (missing(value)) {
        return(self$otimizer$parset)
      } else {
        self$otimizer$parset = assert_class(value, "ParamSet", null.ok = TRUE)
        return(self)
      }
    }
  ),
  private = list(
    # Hyperparameters
    .max_nrounds = 3*10^3L,
    .early_stopping_rounds = 20L,
    .early_stopping_fraction = 4/5,
    .impact_encoding_boundary = 10L,
    .tune_threshold = TRUE,
    .nthread = NULL,
    .resample_instance = NULL,
    .logger = NULL,
    .watch = NULL,
    baselearner = NULL
  )
)
