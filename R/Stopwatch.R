#' @title Control runtime of AutoxgboostMC runs
#' @name Stopwatch
Stopwatch = R6::R6Class("Stopwatch",
  public = list(
    start_time = NULL,
    per_fit_time = NULL,
    time_budget = NULL,
    max_iterations = NULL,
    current_iter = 0L,
    initialize = function(time_budget = NULL, iterations = NULL, start = TRUE) {
      assert_flag(start)
      if (is.null(time_budget) & is.null(iterations))
        stop("At least one of time_budget and iterations must be non-null!")

      self$time_budget = assert_integerish(time_budget, null.ok = TRUE)
      if (is.null(time_budget)) self$time_budget = Inf

      self$max_iterations = assert_integerish(iterations, null.ok = TRUE)
      if (is.null(iterations)) self$max_iterations = Inf
      # FIXME: Write training time  from outside
      if (is.null(self$per_fit_time)) self$set_per_fit_time()
      if (start) self$start()
    },
    get_time_left = function() {
      floor(self$time_budget - as.numeric(Sys.time() - self$start_time))
    },
    set_per_fit_time = function(value = 1L) {
      self$per_fit_time = assert_integerish(value)
    },
    stop = function() {
      time_left = self$get_time_left()
      if (is.numeric(time_left) & time_left >= 0L) {
        (time_left < self$per_fit_time) || (self$current_iter >= self$max_iterations)
      } else {
        TRUE
      }
    },
    start = function() {
      self$start_time = Sys.time()
      self$current_iter = 0L
    },
    reset = function() {
      self$start()
    },
    increment_iter = function(by = 1L) {
      assert_integerish(by)
      self$current_iter = self$current_iter + by
    }
  )
)
