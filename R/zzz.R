#' @import ParamHelpers
#' @import mlr
#' @import mlrMBO
#' @import smoof
#' @import mlrCPO
#' @import xgboost
#' @import BBmisc
#' @import checkmate
#' @import R6
#' @import log4r
#' @importFrom stats predict
#' @importFrom stats runif rnorm setNames aggregate quantile

registerS3method("makeRLearner", "regr.autoxgboostMC", makeRLearner.regr.autoxgboostMC)
registerS3method("trainLearner", "regr.autoxgboostMC", trainLearner.regr.autoxgboostMC)
registerS3method("predictLearner", "regr.autoxgboostMC", predictLearner.regr.autoxgboostMC)

registerS3method("makeRLearner", "classif.autoxgboostMC", makeRLearner.classif.autoxgboostMC)
registerS3method("trainLearner", "classif.autoxgboostMC", trainLearner.classif.autoxgboostMC)
registerS3method("predictLearner", "classif.autoxgboostMC", predictLearner.classif.autoxgboostMC)
