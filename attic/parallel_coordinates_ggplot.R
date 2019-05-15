# assert_choice(type, c("parcoord", "ggplot", "plotly"))
# else p = plot_parallel_coordinates_ggplot(opt_df)
# if (type == "plotly") p = plotly::ggplotly(p, tooltip = "text")

# plot_parallel_coordinates_ggplot = function(opt_df, order) {
#   par_range = sapply(opt_df[, pars], range)
#   normed_pars = t((t(opt_df[, pars]) - par_range[1,]) / (par_range[2,] - par_range[1,]))
#   colnames(opt_df) = paste0("_", pars)
#   opt_df = cbind(opt_df, normed_pars)
#   opt_df$iter = seq_len(nrow(opt_df))
#   pdf_norm = tidyr::gather_(opt_df, key = "normed_x", value = "y", colnames(normed_pars))

#   text = make_parallel_coords_text(opt_df)
#   pdf_norm$tooltip = text[pdf_norm$iter]
#   # Order x-axis and reorder for consitent paths
#   pdf_norm$normed_x = factor(as.character(pdf_norm$normed_x), levels = c(self$measure_ids, names(self$parset$pars)))
#   pdf_norm = pdf_norm[order(pdf_norm$normed_x, decreasing = FALSE),]
#   p = ggplot2::ggplot(pdf_norm, ggplot2::aes(x = normed_x, y = y, group = iter,
#     text = tooltip)) +
#     ggplot2::geom_path(ggplot2::aes(color = iter, alpha = y), alpha = 0.25) +
#     ggplot2::geom_point(size = 4L, alpha = 0.5, color = "grey") +
#     ggplot2::theme_bw() +
#     ggplot2::guides(color = FALSE) +
#     ggplot2::theme(
#       axis.text.y = ggplot2::element_blank(),
#       axis.ticks.y = ggplot2::element_blank(),
#       axis.text.x = ggplot2::element_text(angle = 22, hjust = 1, colour = ifelse(levels(pdf_norm$normed_x) %in% self$measure_ids, "blue", "black"))
#     ) +
#     ggplot2::ylab("") + ggplot2::xlab("")
#   return(p)
# }

# make_parallel_coords_text = function(opt_df) {
#   pdf_text = tidyr::gather_(opt_df, key = "text", value = "textval")
#   text = unlist(sapply(split(pdf_text, pdf_text$iter), function(x) {
#     iter = paste0("Iteration:", unique(pdf_text$iter))
#      # Measures:
#      meas = x[x$text %in% paste0("_", self$measure_ids),]
#      meas = paste0(gsub("_", "", meas$text), ":", round(meas$textval, 3))
#      meas = paste0("<br>Measures:<br>", paste0(unique(meas), collapse = "<br>"))
#      # Parameters
#      pars = x[!(x$text %in% paste0("_", self$measure_ids)), ]
#      pars = paste0(gsub("_", "", pars$text), ":", round(pars$textval, 3))
#      pars = paste0("<br>Parameters:<br>", paste0(unique(pars), collapse = "<br>"))
#      paste0(meas, pars)
#   }))
# }
