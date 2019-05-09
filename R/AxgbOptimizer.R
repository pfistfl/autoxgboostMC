#' Abstract Base Class
AxgbOptimizer = R6::R6Class("AxgbOptimizer",
  public = list(
    fit = function(iterations, time_budget, fit_final_model, plot) {stop("Abstract Base Class")}
  )
)

#' Optimize using SMBO
AxgbOptimizerSMBO = R6::R6Class("AxgbOptimizerSMBO",
  inherits = AxgbOptimizer,
  public = list(
    task = NULL,
    measures = NULL,

    iterations = NULL,
    time_budget = NULL,

    preproc_pipeline = NULL,
    obj_fun = NULL,
    opt_state = NULL,
    opt_result = NULL,
    fit = function(iterations, time_budget, fit_final_model, plot) {
      if (is.null(self$opt_state)) {
        log4r::info(private$.logger, "Evaluating initial design")
        self$opt_state = self$init_smbo()
      }

      log4r::info(private$.logger, "Starting MBO")
      while(!private$.watch$stop()) self$fit_iteration(plot = plot)

      log4r::info(private$.logger, "Finalizing MBO")
      self$opt_result = self$finalize_smbo()
    },
    fit_iteration = function(plot) {
      log4r::debug(private$.logger, catf("Fitting Iteration %s", private$.watch$current_iter))
      prop = proposePoints(self$opt_state)
      x = Map(f = function(par, pt) {
        if(!is.null(par$trafo)) par$trafo(pt)
        else pt
      }, par = self$parset$pars, pt = dfRowsToList(df = prop$prop.points, par.set = self$parset)[[1]])
      y = self$obj_fun(x)
      updateSMBO(self$opt_state, x = prop$prop.points, y = y)
      # Write out .nrounds etc. (currently missing in mlrMBO)
      self$opt_state$opt.path$env$extra[[length(self$opt_state$opt.path$env$extra)]] = c(self$opt_state$opt.path$env$extra[[length(self$opt_state$opt.path$env$extra)]], attr(y, "extras"))
      self$watch$increment_iter()
      if(plot) self$plot_opt_path()
    }),
    init_smbo = function() {
      assert_class(private$.control, "MBOControl", null.ok = TRUE)
      # Set defaults
      if (is.null(private$.control)) {
        private$.control = makeMBOControl(n.objectives = length(self$measures), y.name = self$measure_ids)
        if (self$is_multicrit) {
          private$.control = setMBOControlMultiObj(private$.control, method = "dib", dib.indicator = "eps")
          private$.control = setMBOControlInfill(private$.control, crit = makeMBOInfillCritDIB(cb.lambda = 2L))
        }
      }
      des = generateDesign(n = private$.design_size, private$.parset)
      # Doing one iteration here to evaluate design saves a lot of redundancy.
      private$.control = setMBOControlTermination(private$.control, iters = 1L)
      opt_result = mbo(fun = self$obj_fun, design = des, learner = private$.mbo_learner,
        control = private$.control)
      private$.watch$increment_iter(private$.design_size + 1)
      return(opt_result$final.opt.state)
    },
    finalize_smbo = function() {
      opt_result = finalizeSMBO(self$opt_state)
      if(length(self$measures) > 1L) {
        # Fill best.ind, x and y using "best on early stopping measure".
        opt_result$best.ind = self$get_best_ind(opt_result)
        pars = names(opt_result$opt.path$par.set$pars)
        opt_result$x = as.list(opt_result$opt.path$env$path[self$get_best_ind(opt_result), pars])
        opt_result$y = as.list(opt_result$opt.path$env$path[self$get_best_ind(opt_result), self$measure_ids])
      }
      return(opt_result)
    },
)
