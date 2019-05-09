#' Abstract Base Class
AxgbOptimizer = R6::R6Class("AxgbOptimizer",
  public = list(
    obj_state = NULL,
    obj_result = NULL,
    initialize = function(obj_fun, parset, logger, watch) {
      private$.obj_fun = obj_fun
      private$.parset = parset
      private$.logger = assert_class(logger, "logger")
      private$.watch = watch
    },
    fit = function(iterations, time_budget, plot) {stop("Abstract Base Class")}
  ),
  private = list(
    .watch = NULL,
    .logger = NULL,
    .obj_fun = NULL,
    .parset = NULL
  )
)

#' Optimize using SMBO
AxgbOptimizerSMBO = R6::R6Class("AxgbOptimizerSMBO",
  inherit = AxgbOptimizer,
  public = list(
    initialize = function(obj_fun, parset, logger, watch) {
      private$.obj_fun = assert_class(obj_fun, "smoof_function")
      private$.parset  = assert_class(parset, "ParamSet")
      private$.logger  = assert_class(logger, "logger")
      private$.watch   = assert_class(watch, "Stopwatch")
    },
    fit = function(iterations, time_budget, plot) {
      private$.watch$start()
      if (is.null(self$opt_state)) {
        log4r::info(private$.logger, "Evaluating initial design")
        self$opt_state = private$init_smbo()
      }

      log4r::info(private$.logger, "Starting MBO")
      while(!private$.watch$stop()) private$fit_iteration(plot = plot)

      log4r::info(private$.logger, "Finalizing MBO")
      self$opt_result = private$finalize_smbo()
    }
  ),
  private = list(
    .control = NULL,
    .design_size = 15L,
    .mbo_learner = NULL,
    fit_iteration = function(plot) {
      log4r::debug(private$.logger, catf("Fitting Iteration %s", private$.watch$current_iter))
      prop = proposePoints(self$opt_state)
      x = trafoValue(prop$prop.points, private$.parset)
      y = self$obj_fun(x)
      updateSMBO(self$opt_state, x = prop$prop.points, y = y)
      # Write out .nrounds etc. (currently missing in mlrMBO)
      self$opt_state$opt.path$env$extra[[length(self$opt_state$opt.path$env$extra)]] = c(self$opt_state$opt.path$env$extra[[length(self$opt_state$opt.path$env$extra)]], attr(y, "extras"))
      self$watch$increment_iter()
      if(plot) self$plot_opt_path()
    },
    init_smbo = function() {
      assert_class(private$.control, "MBOControl", null.ok = TRUE)
      # Set defaults
      if (is.null(private$.control)) {
        private$.control = makeMBOControl(n.objectives = length(self$measures), y.name = self$measure_ids)
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
    logger = function() {private$.logger},
    watch = function() {private$.watch},
    obj_fun = function() {private$.obj_fun},
    minimize = function() {
      attr(private$.obj_fun, "minimize")
    },
    n_objectives = function() {
      attr(private$.obj_fun, "n.objectives")
    }
  )
)
