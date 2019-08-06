Sys.setenv("R_TESTS" = "")

library(testthat)
library(checkmate)
library(autoxgboostMC)
library(mlr)

test_check("autoxgboostMC")
