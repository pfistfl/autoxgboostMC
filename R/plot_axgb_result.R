#' Nonconvex pareto front
plot_pareto_front = function(x = NULL, y = NULL, plotly = FALSE) {
  assert(self$is_multicrit)
  assert_flag(plotly)
  df = self$get_opt_path_df()
  # Drop 2nd lambda (MBO param)
  df = df[, -rev(which(colnames(df) == "lambda"))[1]]
  assert_choice(x, self$measure_ids, null.ok = TRUE)
  assert_choice(y, self$measure_ids, null.ok = TRUE)
  if (is.null(x)) x = self$measure_ids[1]
  if (is.null(y) & length(self$measures) >= 2L) y = self$measure_ids[2]
  measures = setNames(self$measures, self$measure_ids)

  front_data = as.data.frame(getOptPathParetoFront(mlrMBO:::getOptStateOptPath(self$optimizer$opt_state)))
  front_data = perf_retrafo_opt_path(front_data, self$measures)
  front = get_pareto_data_nonconvex(front_data, x, y, measures[[x]]$minimize)

  p = ggplot2::ggplot(df, ggplot2::aes_string(x = x, y = y)) +
  ggplot2::geom_point(size = 1.5, alpha = 0.6, color = "red") +
  ggplot2::geom_path(data = front$line, ggplot2::aes_string(x = x, y = y), alpha = 0.7, color = "darkgrey", size = 1) +
  ggplot2::geom_point(data = front$points, alpha = 0.9, size = 1.5, color = "black") +
  ggplot2::theme_bw()

  if (!measures[[x]]$minimize) {p = p + ggplot2::scale_x_reverse()}
  if (!measures[[y]]$minimize) {p = p + ggplot2::scale_y_reverse()}
  if (plotly) plotly::ggplotly(p)
  else p
}

#' Nonconvex pareto front for a given weight range (limiting the range of random projections).
plot_pareto_front_projections = function(x = NULL, y = NULL, wt_range = c(0, 1), plotly = FALSE) {
  assert_choice(x, self$measure_ids, null.ok = TRUE)
  assert_choice(y, self$measure_ids, null.ok = TRUE)
  assert_numeric(wt_range, lower = 0, upper = 1, len = 2L)
  assert_flag(plotly)
  if (sum(wt_range) == 0) error("At least 1 element of wt_range must be > 0!")
  if (is.null(x)) x = self$measure_ids[1]
  if (is.null(y) & length(self$measures) >= 2L) y = self$measure_ids[2]

  measures = setNames(self$measures, self$measure_ids)
  minimize = vlapply(measures[c(x, y)], function(x) x$minimize)

  front_data = as.data.frame(getOptPathParetoFront(mlrMBO:::getOptStateOptPath(self$optimizer$opt_state)))
  front_data = perf_retrafo_opt_path(front_data, self$measures)
  df = get_pareto_data_nonconvex(front_data, x, y, measures[[x]]$minimize)
  points_normal = normalize(df$points, "range")
  best_points = viapply(wt_range, function(wt) {
    wt = c(wt, 1 - wt)
    wt[!minimize] = - wt[!minimize]
    best = which.min(points_normal[, x] * wt[1] + points_normal[, y] * wt[2])
  })

  df_focus = get_pareto_data_nonconvex(df$points[seq(from = best_points[1], to = best_points[2]), ],
    x, y, measures[[x]]$minimize)

  p = self$plot_pareto_front(x, y, plotly = FALSE) +
    ggplot2::geom_point(data = df_focus$points, ggplot2::aes_string(x = x, y = y), color = "blue", shape = 16L, size = 3L, alpha = 0.8) +
    ggplot2::geom_path( data = df_focus$line, ggplot2::aes_string(x = x, y = y)  , color = "blue", size = 1.5, alpha = .6)

  if (plotly) plotly::ggplotly(p)
  else p
}
#' Obtain the data required to plot non-convex pareto fronts
get_pareto_data_nonconvex = function(front_data, x, y, x_minimize) {
  # Make sure data is sorted
  front_data = front_data[order(front_data[[x]], decreasing = x_minimize), c(x, y)]
  # FIXME: This should work for all combinations of measure$minimize
  front_line = data.frame(front_data[[x]][-nrow(front_data)], front_data[[y]][-1])
  colnames(front_line) = colnames(front_data)
  front_line = rbind(front_data, front_line)
  front_line = front_line[order(front_line[[x]], decreasing = x_minimize), ]
  return(list(points = front_data, line = front_line))
}

#' Optimization path
plot_opt_path = function() {
  opt_df = self$get_opt_path_df()
  opt_df = opt_df[!is.na(opt_df$prop.type), ] # Delete Subevals
  opt_df$iter = seq_len(nrow(opt_df))
  cumbest = function(x, minimize) {if(minimize) cummin(x) else cummax(x)}
  measure_minimize = setNames(self$measure_minimize, self$measure_ids)
  pdf = do.call("rbind", lapply(self$measure_ids, function(x)
    data.frame("value" = opt_df[,x], "measure" = x, "iter" = opt_df$iter, "value_opt" = cumbest(opt_df[,x], measure_minimize[x]))))
  p = ggplot2::ggplot(pdf) +
    ggplot2::geom_point(ggplot2::aes(x = iter, y = value, color = measure), alpha = 0.7) +
    ggplot2::geom_path(ggplot2::aes(x = iter, y = value, color = measure), alpha = 0.4) +
    ggplot2::geom_path(ggplot2::aes(x = iter, y = value_opt, color = measure), alpha = 0.6, size = 1.5) +
    ggplot2::theme_bw() +
    ggplot2::ylab("") +
    ggplot2::facet_grid(measure ~ ., scales = "free_y") +
    ggplot2::guides(color = FALSE) +
    ggplot2::scale_color_brewer(palette = "Set1")
  print(p)
}

#' Parallel Coordinates Plot
plot_parallel_coordinates = function(order = "hclust") {
  assert_choice(order, c("none", "hclust"))

  # Get data in correct format
  opt_df = self$get_opt_path_df()
  pars = c(self$measure_ids, names(self$parset$pars))
  opt_df = opt_df[, pars]

  # Reorder using hclust
  # if (order == "hclust") {
  #   hc = hclust(dist(cor(opt_df)))
  #   opt_df = opt_df[hc$order, ]
  # }

  p = plot_parallel_coordinates_parcoords(opt_df)
  return(p)
}


plot_parallel_coordinates_parcoords = function(opt_df, order) {
  requireNamespace("parcoords", quietly = TRUE)
  parcoords::parcoords(opt_df, reorderable = TRUE, brushMode = "2d-strums", alphaOnBrushed = 0.05, rownames = FALSE)
}

