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
      private$.logger = log4r::logger()

      private$baselearner = self$make_baselearner()
      transf_tasks = self$build_transform_pipeline()
      private$baselearner = setHyperPars(private$baselearner, early.stopping.data = transf_tasks$task.test)
      self$obj_fun = self$make_objective_function(transf_tasks)
    },
    print = function(...) {
      catf("AutoxgboostMC Learner")
      catf("Trained: %s", ifelse(is.null(self$model), "no", "yes"))
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

      # Start the Stopwatch
      private$.watch = Stopwatch$new(time_budget, iterations)
      self$fit_smbo(iterations, time_budget, fit_final_model, plot)

      # Create final model
      self$final_learner = self$build_final_learner()
      if(fit_final_model) self$fit_final_model()
    },
    fit_smbo = function(iterations, time_budget, fit_final_model, plot) {
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
    },
    fit_final_model = function() {
      if(is.null(self$final_learner)) self$build_final_learner()
      self$final_model = train(self$final_learner, self$task)
    },
    predict = function(newdata) {
      assert(check_class(newdata, "data.frame"), check_class(newdata, "Task"), combine = "or")
      if(is.null(self$final_model)) stop("Final model not fitted, use .$fit_final_model() to fit!")
      predict(self$final_model, newdata)
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
      lapply(names(par_vals), function(x) self[[x]](par_vals[[x]]))
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


    ## Plot functions -------------------------------------------------------------------
    plot_pareto_front = function(x = NULL, y = NULL, color = NULL, plotly = FALSE) {
      assert(self$is_multicrit)
      df = self$get_opt_path_df()
      # Drop 2nd lambda (MBO param)
      df = df[, -rev(which(colnames(df) == "lambda"))[1]]
      assert_choice(x, self$measure_ids, null.ok = TRUE)
      assert_choice(y, self$measure_ids, null.ok = TRUE)
      assert_choice(color, colnames(df), null.ok = TRUE)
      if (is.null(x)) x = self$measure_ids[1]
      if (is.null(y) & length(self$measures) >= 2L) y = self$measure_ids[2]

      # Plot pareto front
      df$on_front = seq_len(nrow(df)) %in% self$opt_result$pareto.inds
      front_data = df[df$on_front, c(x, y)]
      front_data = front_data[order(front_data[[x]]), ]

      p = ggplot2::ggplot(df, ggplot2::aes_string(x = x, y = y, color = color)) +
      ggplot2::geom_path(data = front_data, ggplot2::aes_string(x = x, y = y), alpha = 0.5, color = "grey", size = 1) +
      ggplot2::geom_point(ggplot2::aes(alpha = on_front, color = on_front), size = 1.5) +
      ggplot2::theme_bw() +
      ggplot2::guides(color = FALSE, alpha = FALSE) +
      ggplot2::scale_color_manual(values = c("red", "black")) +
      ggplot2::scale_alpha_manual(values= c(1, 0.6))

      measures = setNames(self$measures, self$measure_ids)
      if (!measures[[x]]$minimize) {p = p + ggplot2::scale_x_reverse()}
      if (!measures[[y]]$minimize) {p = p + ggplot2::scale_y_reverse()}
      if (plotly) plotly::ggplotly(p)
      else p
    },
    plot_opt_result = function() {
      plot(self$opt_result)
    },
    plot_opt_path = function() {
      opt_df = self$get_opt_path_df()
      opt_df$iter = seq_len(nrow(opt_df))
      cumbest = function(x, minimize) {if(minimize) cummin(x) else cummax(x)}
      measure_minimize = setNames(self$measure_minimize, self$measure_ids)
      pdf = do.call("rbind", lapply(self$measure_ids, function(x)
        data.frame("value" = opt_df[,x], "measure" = x, "iter" = opt_df$iter, "value_opt" = cumbest(opt_df[,x], measure_minimize[x]))))
      p = ggplot2::ggplot(pdf) +
        ggplot2::geom_point(ggplot2::aes(x = iter, y = value, color = measure), alpha = 0.7) +
        ggplot2::geom_path(ggplot2::aes(x = iter, y = value, color = measure), alpha = 0.5) +
        ggplot2::geom_path(ggplot2::aes(x = iter, y = value_opt, color = measure), alpha = 0.8, size = 2L) +
        ggplot2::theme_bw() +
        ggplot2::ylab("") +
        ggplot2::facet_grid(measure ~ ., scales = "free_y") +
        ggplot2::guides(color = FALSE)
      print(p)
    },
    plot_parallel_coordinates = function(trim = 20L) {
      requirePackages("tidyr")
      opt_df = self$get_opt_path_df()
      opt_df = opt_df[opt_df[, self$early_stopping_measure$id] >= sort(opt_df[, self$early_stopping_measure$id], decreasing = TRUE)[min(trim, nrow(opt_df))],]
      # Drop 2nd lambda (MBO param)
      opt_df = opt_df[, -rev(which(colnames(opt_df) == "lambda"))[1]]
      pars = c(names(self$parset$pars), self$measure_ids)
      opt_df = opt_df[, pars]
      par_range = sapply(opt_df[, pars], range)
      normed_pars = t((t(opt_df[, pars]) - par_range[1,]) / (par_range[2,] - par_range[1,]))
      colnames(opt_df) = paste0("_", pars)
      opt_df = cbind(opt_df, normed_pars)
      opt_df$iter = seq_len(nrow(opt_df))
      pdf_norm = tidyr::gather_(opt_df, key = "normed_x", value = "y", colnames(normed_pars))
      pdf_text = tidyr::gather_(opt_df, key = "text", value = "textval", paste0("_", pars))
      text = unlist(sapply(split(pdf_text, pdf_text$iter), function(x) {
        iter = paste0("Iteration:", unique(pdf_text$iter))
         # Measures:
         meas = x[x$text %in% paste0("_", self$measure_ids),]
         meas = paste0(gsub("_", "", meas$text), ":", round(meas$textval, 3))
         meas = paste0("<br>Measures:<br>", paste0(unique(meas), collapse = "<br>"))
         # Parameters
         pars = x[!(x$text %in% paste0("_", self$measure_ids)), ]
         pars = paste0(gsub("_", "", pars$text), ":", round(pars$textval, 3))
         pars = paste0("<br>Parameters:<br>", paste0(unique(pars), collapse = "<br>"))
         paste0(meas, pars)
      }))
      pdf_norm$tooltip = text[pdf_norm$iter]
      # Order x-axis and reorder for consitent paths
      pdf_norm$normed_x = factor(as.character(pdf_norm$normed_x), levels = c(self$measure_ids, names(self$parset$pars)))
      pdf_norm = pdf_norm[order(pdf_norm$normed_x, decreasing = FALSE),]
      p = ggplot2::ggplot(pdf_norm, ggplot2::aes(x = normed_x, y = y, group = iter,
        text = tooltip)) +
        ggplot2::geom_path(ggplot2::aes(color = iter, alpha = y), alpha = 0.25) +
        ggplot2::geom_point(size = 4L, alpha = 0.5, color = "grey") +
        ggplot2::theme_bw() +
        ggplot2::guides(color = FALSE) +
        ggplot2::theme(
          axis.text.y = ggplot2::element_blank(),
          axis.ticks.y = ggplot2::element_blank(),
          axis.text.x = ggplot2::element_text(angle = 22, hjust = 1, colour = ifelse(levels(pdf_norm$normed_x) %in% self$measure_ids, "blue", "black"))
        ) +
        ggplot2::ylab("") + ggplot2::xlab("")

      gg = plotly::ggplotly(p, tooltip = "text")
      plotly::highlight(gg, dynamic = TRUE)
    }
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
    # MBO Hyperparameters
    control = function(value) {
      if (missing(value)) {
        return(private$.control)
      } else {
        private$.control = assert_class(value, "MBOControl")
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
    parset = function(value) {
      if (missing(value)) {
        return(private$.parset)
      } else {
        private$.parset = assert_class(value, "ParamSet", null.ok = TRUE)
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
    watch = function() {private$.watch}
  ),
  private = list(
    # Hyperparameters
    .max_nrounds = 3*10^3L,
    .early_stopping_rounds = 20L,
    .early_stopping_fraction = 4/5,
    .impact_encoding_boundary = 10L,
    .tune_threshold = TRUE,
    .nthread = NULL,

    # MBO Hyperparameters
    .control = NULL,
    .parset = NULL,
    .design_size = 15L,
    .mbo_learner = NULL,
    .resample_instance = NULL,

    .logger = NULL,
    .watch = NULL,
    baselearner = NULL
  )
)
