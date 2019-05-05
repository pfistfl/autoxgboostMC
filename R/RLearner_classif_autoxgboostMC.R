#' @export
makeRLearner.classif.autoxgboostMC = function() {
  makeRLearnerClassif(
    cl = "classif.autoxgboostMC",
    package = "autoxgboostMC",
    par.set = makeParamSet(
      makeUntypedLearnerParam(id = "measure", default = mse),
      makeUntypedLearnerParam(id = "control"),
      makeIntegerLearnerParam(id = "iterations", lower = 1, default = 160),
      makeIntegerLearnerParam(id = "time.budget", lower = 1, default = 3600),
      makeUntypedLearnerParam(id = "par.set", default = autoxgbparset),
      makeIntegerLearnerParam(id = "max.nrounds", lower = 1L, default = 10L^6),
      makeIntegerLearnerParam(id = "early.stopping.rounds", lower = 1, default = 10L),
      makeNumericLearnerParam(id = "early.stopping.fraction", lower = 0, upper = 1, default = 4/5),
      makeIntegerLearnerParam(id = "design.size", lower = 1L, default = 15L),
      makeDiscreteLearnerParam(id  = "factor.encode", values = c("impact", "dummy"), default = "impact"),
      makeUntypedLearnerParam(id = "mbo.learner"),
      makeIntegerLearnerParam(id = "nthread", lower = 1L, tunable = FALSE),
      makeLogicalLearnerParam(id = "tune.threshold", default = TRUE)
    ),
    properties = c("twoclass", "multiclass", "numerics", "factors", "prob", "missings"),
    name = "Automatic  multi-criteria eXtreme Gradient Boosting",
    short.name = "autoxgboostMC",
    note = ""
    )
}

#' @export
trainLearner.classif.autoxgboostMC = function(.learner, .task, .subset, .weights = NULL,
  measures = list(mmce), control = NULL, iterations = 160, time.budget = 30, par.set = autoxgbparset, max.nrounds = 10^6, early.stopping.rounds = 10L,
  early.stopping.fraction = 4/5, build.final.model = TRUE, design.size = 15L,
  impact.encoding.boundary = 10L, mbo.learner = NULL, nthread = NULL, tune.threshold = TRUE, ...) {

  .task = subsetTask(.task, .subset)
  axgb = AutoxgboostMC$new(.task, measures = measures)
  axgb$fit(time.budget = time.budget, iterations = iteratinos)
  return(axgb)
}

#' @export
predictLearner.classif.autoxgboostMC = function(.learner, .model, .newdata, ...) {
  .model$learner.model$predict(.newdata)
}
