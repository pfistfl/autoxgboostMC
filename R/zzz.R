#' @import mlr
#' @import ParamHelpers
#' @import smoof
#' @import mlrMBO
#' @import xgboost
#' @import BBmisc
#' @import checkmate
#' @import mlrCPO
#' @importFrom stats predict
#' @importFrom stats runif
#' @importFrom stats aggregate


registerS3method("makeRLearner", "regr.autoxgboostMC", makeRLearner.regr.autoxgboost)
registerS3method("trainLearner", "regr.autoxgboostMC", trainLearner.regr.autoxgboost)
registerS3method("predictLearner", "regr.autoxgboostMC", predictLearner.regr.autoxgboost)

registerS3method("makeRLearner", "classif.autoxgboostMC", makeRLearner.classif.autoxgboost)
registerS3method("trainLearner", "classif.autoxgboostMC", trainLearner.classif.autoxgboost)
registerS3method("predictLearner", "classif.autoxgboostMC", predictLearner.classif.autoxgboost)
