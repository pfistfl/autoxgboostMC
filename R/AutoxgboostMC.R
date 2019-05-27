#' @title Fit and optimize a xgboost model for multiple criteria
#'
#' @include AxgbOptimizer.R
#' @include AxgbPipeline.R
#' @include plot_axgb_result.R
#' @include helpers.R
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
#' The optimization process can be controlled via additional arguments to `.$optimizer`.
#' See `\code{\link{AxgbOptimizer}} for more information.
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

    pipeline = NULL,
    obj_fun = NULL,

    optimizer = NULL,
    iterations = NULL,
    time_budget = NULL,

    final_learner = NULL,
    final_model = NULL,

    initialize = function(task, measures = NULL, parset = NULL, nthread = NULL,
      pipeline  = AxgbPipelineXGB$new(),
      optimizer = AxgbOptimizerSMBO$new()
      ) {

      self$task = assert_class(task, "SupervisedTask")
      assert_list(measures, types = "Measure", null.ok = TRUE)
      assert_class(parset, "ParamSet", null.ok = TRUE)

      # Set defaults
      self$measures = coalesce(measures, list(getDefaultMeasure(task)))
      private$.logger = log4r::logger(threshold = "WARN")

      # Construct pipeline and learner
      self$pipeline = assert_class(pipeline, "AxgbPipeline")
      self$pipeline$configure(logger = private$.logger, parset = parset)
      obj_fun = self$pipeline$get_objfun(self$task, self$measures, parset, nthread)

      # Initialize Optimizer
      self$optimizer = assert_class(optimizer, "AxgbOptimizer")
      self$optimizer$configure(measures = self$measures, obj_fun = obj_fun, parset = attr(obj_fun, "par.set"), logger = private$.logger)
    },
    fit = function(iterations = 160L, time_budget = 3600L, fit_final_model = FALSE, plot = TRUE) {
      assert_integerish(iterations)
      assert_integerish(time_budget)
      assert_flag(fit_final_model)
      assert_flag(plot)

      # Initialize  the optimizer
      self$optimizer$fit(iterations, time_budget, plot)

      # Create final model
      self$final_learner = self$pipeline$build_final_learner(self$optimizer$get_opt_pars())
      if(fit_final_model) self$fit_final_model()
    },
    predict = function(newdata) {
      assert(check_class(newdata, "data.frame"), check_class(newdata, "Task"), combine = "or")
      if(is.null(self$final_model)) stop("Final model not fitted, use .$fit_final_model() to fit!")
      predict(self$final_model, newdata)
    },
    print = function(...) {
      catf("AutoxgboostMC Learner")
      catf("Task: %s (%s)", self$task$task.desc$id, self$task$type)
      catf("Measures: %s", paste0(self$measure_ids, collapse = ","))
      catf("Trained: %s", ifelse(is.null(self$opt_result), "no", "yes"))
      print(self$optimizer)
      catf("\n\nPreprocessing pipeline:")
      print(self$pipeline$preproc_pipeline)
    },
    fit_final_model = function() {
      if(is.null(self$final_learner)) self$pipeline$build_final_learner()
      self$final_model = train(self$final_learner, self$task)
    },

    ## Setters for various hyperparameters -----------------------------------------------
    set_hyperpars = function(par_vals) {
      assert_list(par_vals, names = "unique")
      lapply(names(par_vals), function(x) self[[x]] = par_vals[[x]])
      invisible(self)
    },
    set_parset_bounds = function(param, lower = NULL, upper = NULL) {
      ps = self$parset
      assert_choice(param, names(ps$pars))
      if(!is.null(lower)) ps$pars[[param]]$lower = assert_number(lower)
      if(!is.null(upper)) ps$pars[[param]]$upper = assert_number(upper)
      self$parset = ps
      invisible(self)
    },
    plot_pareto_front = plot_pareto_front,
    plot_pareto_front_projections = plot_pareto_front_projections,
    plot_parallel_coordinates = plot_parallel_coordinates,
    plot_opt_result = function() {self$optimizer$plot_opt_result()},
    plot_opt_path = function() {self$optimizer$plot_opt_path()},
    get_opt_path_df = function() {self$optimizer$get_opt_path_df()}
  ),

  active = list(
    # AB for optimizer --------------------------------------------------------------------
    parset = function(value) {
      if (missing(value)) {
        attr(self$optimizer$obj_fun, "par.set" )
      } else {
        attr(self$optimizer$obj_fun, "par.set" ) = assert_class(value, "ParamSet", null.ok = TRUE)
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

    # Hyperparameters for the Pipeline Class -------------------------------------
    logger = function(value) {
        if (missing(value)) {
        return(private$.logger)
      } else {
        assert_class(value, "logger")
        # Push the logger to all fields
        private$.logger = value
        self$pipeline$logger = value
        self$optimizer$logger = value
        return(self)
      }
    },
    watch = function() {private$.watch}
  ),
  private = list(
    .logger = NULL,
    .watch = NULL
  )
)

