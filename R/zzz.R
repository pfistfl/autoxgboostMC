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


registerS3method("makeRLearner", "regr.autoxgboostMC", makeRLearner.regr.autoxgboostMC)
registerS3method("trainLearner", "regr.autoxgboostMC", trainLearner.regr.autoxgboostMC)
registerS3method("predictLearner", "regr.autoxgboostMC", predictLearner.regr.autoxgboostMC)

registerS3method("makeRLearner", "classif.autoxgboostMC", makeRLearner.classif.autoxgboostMC)
registerS3method("trainLearner", "classif.autoxgboostMC", trainLearner.classif.autoxgboostMC)
registerS3method("predictLearner", "classif.autoxgboostMC", predictLearner.classif.autoxgboostMC)
