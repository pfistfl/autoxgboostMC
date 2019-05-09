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
}

#' Optimization path
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
}



plot_parallel_coordinates = function(trim = 20L) {
  requirePackages("tidyr")
  opt_df = self$get_opt_path_df()
  opt_df = opt_df[opt_df[, self$early_stopping_measure$id] >= sort(opt_df[, self$early_stopping_measure$id], decreasing = TRUE)[min(trim, nrow(opt_df))],]
  # Drop 2nd lambda (MBO param)
  # opt_df = opt_df[, -rev(which(colnames(opt_df) == "lambda"))[1]]
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