#' @title Abstract Base Class
#' @export
AxgbOptimizer = R6::R6Class("AxgbOptimizer",
  public = list(
    opt_state = NULL,
    opt_result = NULL,
    fit = function(iterations, time_budget, plot) {stop("Abstract Base Class")},
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
    .watch = NULL,
    .logger = NULL,
    .obj_fun = NULL,
    .parset = NULL
  )
)

#' @title Optimize using SMBO
#'
#' @include plot_axgb_result.R
#' @include helpers.R
#' @export
AxgbOptimizerSMBO = R6::R6Class("AxgbOptimizerSMBO",
  inherit = AxgbOptimizer,
  public = list(
    initialize = function(measures, obj_fun, parset, logger) {
      private$.measures = measures
      private$.obj_fun = assert_class(obj_fun, "smoof_function")
      assert_true(length(measures) == self$n_objectives)
      private$.parset  = assert_class(parset, "ParamSet")
      private$.logger  = assert_class(logger, "logger")
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
    .measures = NULL,
    fit_iteration = function(plot) {
      log4r::debug(private$.logger, catf("Fitting Iteration %s", private$.watch$current_iter))
      prop = proposePoints(self$opt_state)
      x = trafoValue(private$.parset, dfRowsToList(df = prop$prop.points, par.set = self$parset)[[1]])
      y = private$.obj_fun(x)
      updateSMBO(self$opt_state, x = prop$prop.points, y = y)
      # Write out .nrounds etc. (currently missing in mlrMBO)
      # self$opt_state$opt.path$env$extra[[length(self$opt_state$opt.path$env$extra)]] = c(self$opt_state$opt.path$env$extra[[length(self$opt_state$opt.path$env$extra)]], attr(y, "extras"))
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
      des = generateDesign(n = private$.design_size, private$.parset)
      # Doing one iteration here to evaluate design saves a lot of redundancy.
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