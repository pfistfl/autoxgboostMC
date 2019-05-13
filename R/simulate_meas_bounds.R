df = data.frame(
  mmce = c(0.1, 0.09, 0.12),
  pft = c(0.8, 0.9, 0.7)
)

library(ggplot2)
p = ggplot(df, aes(mmce, pft)) +
  geom_point()

normalize = function(mat) {
  apply(mat, 2, function(x) {(x - min(x)) / (max(x) - min(x))})
}
project = function(mat, wt) {
  normalize(mat) %*% c(wt, 1 - wt)
}
project(as.matrix(df), .8)

df_test = data.frame(
  mmce = runif(1000, 0.09, 0.12),
  pft = runif(1000, 0.7, 0.9)
)
score1 = project(as.matrix(df_test), .8)
score2 = project(as.matrix(df_test), .2)
df_test = cbind(df_test, score1, score2)

df_test = df_test[df_test$score1 < 0.6, ]
df_test = df_test[df_test$score2 < 0.6, ]

df_test = df_test[df_test$score1 > 0.2, ]
df_test = df_test[df_test$score2 > 0.2, ]

df_test = df_test[df_test$mmce < 0.115, ]
df_test = df_test[df_test$pft < 0.83, ]
p +
  geom_point(data = df_test,
   aes(mmce, pft)) +
  geom_abline(slope = -4, intercept = 1.2)

