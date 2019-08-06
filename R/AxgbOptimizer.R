#' @title AutoxgboostMC Optimizer Class
#' @format [R6::R6Class]
#' @export
#' @seealso \code{\link{AxgbOptimizerSMBO}}
AxgbOptimizer = R6::R6Class("AxgbOptimizer",
  public = list(
    fit = function() {stop("Abstract Base Class")},
    configure = function(measures, obj_fun, parset, logger) {
      private$.measures = measures
      private$.obj_fun = assert_function(obj_fun)
      private$.parset  = assert_class(parset, "ParamSet")
      private$.logger  = assert_class(logger, "logger")
    },
    print = function(...) {
      if (!is.null(self$opt_result)) {
        op = self$opt_result$opt.path
        pars = trafoValue(op$par.set, self$opt_result$x)
        pars$nrounds = self$get_best("nrounds")
        catf("AutoxgboostMC tuning result")
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
        # FIXME: Nice Printer for results:
        catf("\nWith tuning result:")
        for (i in seq_along(self$measures)) catf("    %s = %.3f", self$measures[[i]]$id, self$opt_result$y[[i]])
        thr = self$get_best(".threshold")
        if (!is.null(thr)) {
          if (length(thr) == 1) {
            catf("\nClassification Threshold: %.3f", thr)
          } else {
            catf("\nClassification Thresholds: %s", paste(names(thr), round(thr, 3), sep = ": ", collapse = "; "))
          }
        }
      }
    }
  ),
  private = list(
    .measures = NULL,
    .logger = NULL,
    .obj_fun = NULL,
    .parset = NULL,
    .watch = NULL
  )
)

#' @title AutoxgboostMC Optimizer using SMBO
#' @format [R6::R6Class] object inheriting from [AxgbOptimizer].
#'
#' @section Construction:
#'  ```
#'  AxgbOptimizerSMBO$new()
#'  ```
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
#' @usage NULL
#' @include plot_axgb_result.R
#' @include helpers.R
#' @export
AxgbOptimizerSMBO = R6::R6Class("AxgbOptimizerSMBO",
  inherit = AxgbOptimizer,
  public = list(
    opt_state = NULL,
    opt_result = NULL,
    initialize = function() {
    },
    fit = function(iterations, time_budget, plot) {
      private$.watch = Stopwatch$new(time_budget, iterations)
      if (is.null(self$opt_state)) {
        log4r::info(private$.logger, "Evaluating initial design")
        self$opt_state = private$init_smbo()
      }

      log4r::info(private$.logger, "Starting MBO")
      while(!private$.watch$stop()) private$fit_iteration(plot = plot)

      log4r::info(private$.logger, "Finalizing MBO")
      self$opt_result = private$finalize_smbo()
    },
    get_opt_pars = function() {
      assert_true(!is.null(self$opt_result))
      pars = trafoValue(self$parset, self$opt_result$x)
      pars = pars[!vlapply(pars, is.na)]
      return(pars)
    },
    set_possible_projections = function(measure_weights) {
      if(self$n_objectives == 2L) {
        assert_numeric(measure_weights, len = 2, lower = 0, upper = 1)
        measure_weights = matrix(measure_weights, nrow = 1L)
      } else {
        assert_matrix(measure_weights, nrows = self$n_objectives - 1,
          ncols = self$n_objectives - 1, mode = "numeric")
      }
      opt_problem = mlrMBO:::getOptStateOptProblem(self$opt_state)
      # Generate possible_weights matrix (this specifies the range
      # of allowed projections
      ncomb = ceiling(100000^(1 / self$n_objectives))
      ncomb = ncomb * 1 / min(1, min(abs(apply(measure_weights, 1, diff)))) # scale ncomb by range between weights
      possible_weights = mlrMBO:::combWithSum(ncomb, self$n_objectives) / ncomb
      # Reorder weights
      vars = apply(possible_weights, 1, var)
      possible_weights = rbind(diag(self$n_objectives), possible_weights[!vars == max(vars),])
      # Force in bisector
      possible_weights = rbind(possible_weights, rep(1/self$n_objectives, self$n_objectives))

      # Only keep allowed projections (by limiting the n_objectives -1 measures).
      keep_weights = sapply(seq_len(self$n_objectives - 1L), function(i) {
        wt = measure_weights[i, ]
        possible_weights[, i] >= min(wt) & possible_weights[, i] <= max(wt)
      })
      possible_weights = possible_weights[apply(keep_weights, 1, all),]
      mlrMBO:::setOptProblemAllPossibleWeights(opt_problem, possible_weights)
    },
    plot_opt_path = plot_opt_path,
    plot_opt_result = function() {
      plot(self$opt_result)
    },
    get_opt_path_df = function() {
      as.data.frame(mlrMBO:::getOptStateOptPath(self$opt_state))
    },
    get_best = function(what) {self$opt_result$opt.path$env$extra[[self$get_best_ind(self$opt_result)]][[what]]},
    get_best_ind = function(opt_result) {
      if (self$early_stopping_measure$minimize) {
        best.ind = which.min(opt_result$opt.path$env$path[[self$early_stopping_measure$id]])
      } else {
        best.ind = which.max(opt_result$opt.path$env$path[[self$early_stopping_measure$id]])
      }
      return(best.ind)
    }
  ),
  private = list(
    .control = NULL,
    .design_size = 15L,
    .mbo_learner = NULL,
    fit_iteration = function(plot, subevals = TRUE) {
      log4r::debug(private$.logger, catf("Fitting Iteration %s", private$.watch$current_iter))
      prop = proposePoints(self$opt_state)
      rownames(prop$prop.points) = NULL
      x = trafoValue(private$.parset, dfRowsToList(df = prop$prop.points, par.set = self$parset)[[1]])
      y = private$.obj_fun(x, subevals = TRUE)

      if (subevals) {
        xy_pareto = get_subevals(prop, y)
        if (!is.null(xy_pareto)) {
          if (length(y) >= 2L) {
            xy_pareto = get_pareto_set(self$opt_state, xy_pareto, private$.parset, private$.measures)
          } else {
            xy_pareto = get_univariate_set(self$opt_state, xy_pareto, private$.measures)
          }
          if (length(xy_pareto$y) > 0) {
            updateSMBO(self$opt_state, x = rbind(prop$prop.points, xy_pareto$x), y = c(list(y), xy_pareto$y))
          }
        }
      } else updateSMBO(self$opt_state, x = prop$prop.points, y = y)
      self$watch$increment_iter()
      if(plot) self$plot_opt_path()
    },
    init_smbo = function() {
      assert_class(private$.control, "MBOControl", null.ok = TRUE)
      # Set defaults
      if (is.null(private$.control)) {
        private$.control = makeMBOControl(n.objectives = self$n_objectives, y.name = self$measure_ids)
        if (self$n_objectives > 1L) {
          private$.control = setMBOControlMultiObj(private$.control, method = "dib", dib.indicator = "eps")
          private$.control = setMBOControlInfill(private$.control, crit = makeMBOInfillCritDIB(cb.lambda = 2L))
        }
      }

      # Get fast initial models
      init_parset = private$.parset
      init_parset$pars$nrounds$upper = 15L
      des = generateDesign(n = private$.design_size, init_parset)

      # Doing one iteration here to evaluate design, saves a lot of redundancy.
      private$.control = setMBOControlTermination(private$.control, iters = 1L)
      opt_result = mbo(fun = private$.obj_fun, design = des, learner = private$.mbo_learner,
        control = private$.control)
      private$.watch$increment_iter(private$.design_size + 1)
      return(opt_result$final.opt.state)
    },
    finalize_smbo = function() {
      opt_result = finalizeSMBO(self$opt_state)
      if (self$n_objectives > 1L) {
        # Fill best.ind, x and y using "best on early stopping measure".
        opt_result$best.ind = self$get_best_ind(opt_result)
        pars = names(opt_result$opt.path$par.set$pars)
        opt_result$x = as.list(opt_result$opt.path$env$path[self$get_best_ind(opt_result), pars])
        opt_result$y = as.list(opt_result$opt.path$env$path[self$get_best_ind(opt_result), self$measure_ids])
      }
      return(opt_result)
    }
  ),
  active = list(
    parset = function(value) {
      if (missing(value)) {
        return(private$.parset)
      } else {
        private$.parset = assert_class(value, "ParamSet", null.ok = TRUE)
        return(self)
      }
    },
    control = function(value) {
      if (missing(value)) {
        return(private$.control)
      } else {
        private$.control = assert_class(value, "MBOControl")
        return(self)
      }
    },
    mbo_learner = function(value) {
      if (missing(value)) {
        return(private$.mbo_learner)
      } else {
        private$.mbo_learner = assert_class(value, "Learner", null.ok = TRUE)
        return(self)
      }
    },
    design_size = function(value) {
      if (missing(value)) {
        return(private$.design_size)
      } else {
        private$.design_size = assert_integerish(value, lower = 1L, len = 1L)
        return(self)
      }
    },
    measures = function() {private$.measures},
    logger = function() {private$.logger},
    watch = function() {private$.watch},
    obj_fun = function() {private$.obj_fun},
    measure_minimize = function() {
      attr(private$.obj_fun, "minimize")
    },
    measure_ids = function() {
      sapply(private$.measures, function(x) x$id)
    },
    n_objectives = function() {
      attr(private$.obj_fun, "n.objectives")
    },
    early_stopping_measure = function(value) {
      if (missing(value)) {
        self$measures[[1]]
      } else {
        measure_ids = sapply(self$measures, function(x)  x$id)
        assert_list(value, types = "Measure", null.ok = TRUE)
        self$measures = c(value, self$measures[-which(value$id == measure_ids)])
        messagef("Setting %s as early stopping measure!", value$id)
      }
    }
  )
)

#' Extract sub-evaluations from the data.frame
get_subevals = function(prop, y) {
  subevals = attr(y, "extra")$.subevals
  if (!is.null(subevals$y)) {
    x = prop$prop.points
    x$nrounds = x$threshold = NULL
    x = cbind(x, subevals$x)
    if (is.null(dim(subevals$y))) subevals$y = data.frame(subevals$y)
    return(list(x = x[, colnames(prop$prop.points)], y = convertRowsToList(subevals$y)))
  } else {
    return(NULL)
  }
}
