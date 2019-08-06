createDMatrixFromTask = function(task, weights = NULL) {
  data = getTaskData(task, target.extra = TRUE)
  data$data = convertDataFrameCols(data$data, ints.as.num = TRUE)
  if (getTaskType(task) == "classif")  {
    cl = getTaskClassLevels(task)
    data$target =  match(as.character(data$target), cl) - 1
  }

  if (!is.null(weights))
    xgboost::xgb.DMatrix(data = data.matrix(data$data), label = data$target, weight = weights)
  else if (!is.null(task$weights))
    xgboost::xgb.DMatrix(data = data.matrix(data$data), label = data$target, weight = task$weights)
  else
    xgboost::xgb.DMatrix(data = data.matrix(data$data), label = data$target)
}

#' Get best nroudns from a model
get_best_iteration = function(mod) {
  getLearnerModel(mod, more.unwrap = TRUE)$best_iteration
}

# Obtain the points from a set xy that are on the pareto front
# given earlier evaluation contained in an opt_state
get_pareto_set = function(opt_state, xy, ps, measures) {
  minimize = vlapply(measures, function(x) x$minimize)
  y.names = vcapply(measures, function(x) x$id)
  op = makeOptPathDF(par.set = ps, y.names = y.names, minimize = minimize)
  for (i in seq_len(nrow(xy$x))) {
    addOptPathEl(op = op, x = convertRowsToList(xy$x)[[i]], y = xy$y[[i]])
  }
  odf = as.data.frame(opt_state$opt.path)
  for (j in seq_len(nrow(odf))) {
    addOptPathEl(op = op, x = convertRowsToList(odf[, colnames(xy$x)])[[j]], y = unlist(odf[j, y.names]))
  }
  idx = intersect(seq_len(nrow(xy$x)), getOptPathParetoFront(op, index = TRUE))
  list(x = xy$x[idx, ], y = xy$y[idx])
}


# Obtain the points from a set xy that are in the 90% quantile
# given earlier evaluation contained in an opt_state.
get_univariate_set = function(opt_state, xy, measures) {
  minimize = vlapply(measures, function(x) x$minimize)
  y.names = vcapply(measures, function(x) x$id)
  odf = as.data.frame(opt_state$opt.path)
  if (minimize) {
    idx = which(xy$y <= quantile(c(odf[, y.names], unlist(xy$y)), 0.1))
  } else {
    idx = which(xy$y >= quantile(c(odf[, y.names], unlist(xy$y)), 0.9))
  }
  idx = intersect(seq_len(nrow(xy$x)), idx)
  list(x = xy$x[idx, ], y = xy$y[idx])
}


# This generates a preprocessing pipeline to handle categorical features
# @param task: the task
# @param impact.encoding.boundary: See autoxgboost
# @return CPOpipeline to transform categorical features
generateCatFeatPipeline = function(task, impact.encoding.boundary) {

  cat.pipeline = cpoFixFactors()

  d = getTaskData(task, target.extra = TRUE)$data
  feat.cols = colnames(d)[vlapply(d, is.factor)]
  #categ.featureset = task$feature.information$categ.featureset
  #if (!is.null(categ.featureset)) {
  #  for(cf in categ.featureset)
  #    cat.pipeline %<>>% cpoFeatureHashing(affect.names = cf)
  # feat.cols = setdiff(feat.cols, unlist(categ.featureset))
  #}

  impact.cols = colnames(d)[vlapply(d, function(x) is.factor(x) && nlevels(x) > impact.encoding.boundary)]
  dummy.cols = setdiff(feat.cols, impact.cols)

  if (length(dummy.cols) > 0L)
      cat.pipeline %<>>% cpoDummyEncode(affect.names = dummy.cols)
  if (length(impact.cols) > 0L) {
    if (getTaskType(task) == "classif") {
      cat.pipeline %<>>% cpoImpactEncodeClassif(affect.names = impact.cols)
    } else {
      cat.pipeline %<>>% cpoImpactEncodeRegr(affect.names = impact.cols)
    }
  }
  return(cat.pipeline)
}

perf_trafo_minimize = function(perf, measures) {
  expect_list(measures)
  expect_numeric(perf)
  minimize = setNames(vlapply(measures, function(x) x$minimize), vcapply(measures, function(x) x$id))
  do_maximize = names(minimize[!minimize])
  if (length(do_maximize) > 0) perf[do_maximize] = - perf[do_maximize]
  return(perf)
}

perf_retrafo = function(x, measure) {
  expect_list(measures)
  expect_numeric(x)
  if (!measure$minimize) x = -x
  return(x)
}

perf_retrafo_opt_path = function(opdf, measures) {
  expect_list(measures)
  expect_data_frame(opdf)
  minimize = setNames(vlapply(measures, function(x) x$minimize), vcapply(measures, function(x) x$id))
  do_maximize = names(minimize[!minimize])
  opdf[, do_maximize] = - opdf[, do_maximize]
  return(opdf)
}
