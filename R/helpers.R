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
