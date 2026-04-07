# ============================================================================
# 03_arima.R
# ARIMA models for Canada and US inflation without explanatory variables
# Tests candidate ARIMA orders and uses best for holdout forecasting
# Output: Tables and diagnostic plots saved to output/tables and output/figures
# ============================================================================

suppressPackageStartupMessages({
  library(tidyverse)
  library(forecast)
})

options(dplyr.summarise.inform = FALSE)

# ============================================================================
# 1. Setup and load data
# ============================================================================
data_processed_dir <- file.path(getwd(), "data_processed")
output_tables_dir <- file.path(getwd(), "output", "tables")
output_figures_dir <- file.path(getwd(), "output", "figures")

dir.create(output_tables_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(output_figures_dir, recursive = TRUE, showWarnings = FALSE)

load(file.path(data_processed_dir, "wrangled_data.RData"))

# ============================================================================
# 2. Data split
# ============================================================================
h <- 24

n_can <- nrow(canada_common)
train_can <- canada_common[1:(n_can - h), ]
test_can  <- canada_common[(n_can - h + 1):n_can, ]

n_us <- nrow(us_common)
train_us <- us_common[1:(n_us - h), ]
test_us  <- us_common[(n_us - h + 1):n_us, ]

# ============================================================================
# 3. Helper functions
# ============================================================================
rmse <- function(actual, pred) {
  sqrt(mean((actual - pred)^2, na.rm = TRUE))
}

# 1-step ahead expanding window ARIMA forecast
arima_expanding_forecast <- function(train, test, order, series_name) {
  cat("ARIMA(", order[1], ",", order[2], ",", order[3], ") for", series_name, "...\n")
  
  preds <- numeric(nrow(test))
  y <- train$infl
  
  for (i in seq_len(nrow(test))) {
    fit <- Arima(y, order = order)
    pred <- as.numeric(forecast::forecast(fit, h = 1)$mean[1])
    preds[i] <- pred
    y <- c(y, test$infl[i])
  }
  
  preds
}

# ============================================================================
# 4. ACF/PACF diagnostics on training set
# ============================================================================
cat("\n=== ACF/PACF DIAGNOSTICS (Training Set) ===\n\n")

png(
  file.path(output_figures_dir, "arima_acf_pacf_diagnostics.png"),
  width = 1000, 
  height = 1100,   
  res = 150,      
  pointsize = 11   
)

par(
  mfrow = c(2, 2), 
  mar = c(6, 5, 5, 2), 
  oma = c(1, 1, 4, 1), 
  cex.main = 1.1, 
  cex.lab = 1.0
)

acf(train_can$infl, main = "Canada Inflation ACF", lag.max = 30)
pacf(train_can$infl, main = "Canada Inflation PACF", lag.max = 30)

acf(train_us$infl, main = "US Inflation ACF", lag.max = 30)
pacf(train_us$infl, main = "US Inflation PACF", lag.max = 30)

mtext("Inflation Diagnostic Plots", outer = TRUE, cex = 1.2, font = 2)

dev.off()
cat("Saved ACF/PACF plot\n")

# ============================================================================
# 5. Candidate ARIMA models
# ============================================================================
cat("\n=== TESTING ARIMA CANDIDATES ===\n\n")

arima_candidates <- list(
  c(1, 0, 1),
  c(2, 0, 1),
  c(1, 0, 2),
  c(1, 1, 1),
  c(2, 1, 1)
)

# ============================================================================
# 6. Find best ARIMA for Canada
# ============================================================================
cat("Testing ARIMA candidates for Canada...\n")

can_arima_rmse <- numeric(length(arima_candidates))
names(can_arima_rmse) <- sapply(arima_candidates, function(o) paste(o, collapse = ","))

for (i in seq_along(arima_candidates)) {
  preds <- arima_expanding_forecast(train_can, test_can, arima_candidates[[i]], "Canada")
  can_arima_rmse[i] <- rmse(test_can$infl, preds)
}

best_can_idx <- which.min(can_arima_rmse)
best_can_order <- arima_candidates[[best_can_idx]]
best_can_preds <- arima_expanding_forecast(train_can, test_can, best_can_order, "Canada")

cat("\nCanada ARIMA Results:\n")
print(tibble(Order = names(can_arima_rmse), RMSE = can_arima_rmse) %>% arrange(RMSE))
cat("Best order for Canada: (", paste(best_can_order, collapse = ","), ")\n")
cat("Holdout RMSE:", can_arima_rmse[best_can_idx], "\n\n")

# ============================================================================
# 7. Find best ARIMA for US
# ============================================================================
cat("Testing ARIMA candidates for US...\n")

us_arima_rmse <- numeric(length(arima_candidates))
names(us_arima_rmse) <- sapply(arima_candidates, function(o) paste(o, collapse = ","))

for (i in seq_along(arima_candidates)) {
  preds <- arima_expanding_forecast(train_us, test_us, arima_candidates[[i]], "United States")
  us_arima_rmse[i] <- rmse(test_us$infl, preds)
}

best_us_idx <- which.min(us_arima_rmse)
best_us_order <- arima_candidates[[best_us_idx]]
best_us_preds <- arima_expanding_forecast(train_us, test_us, best_us_order, "United States")

cat("\nUS ARIMA Results:\n")
print(tibble(Order = names(us_arima_rmse), RMSE = us_arima_rmse) %>% arrange(RMSE))
cat("Best order for US: (", paste(best_us_order, collapse = ","), ")\n")
cat("Holdout RMSE:", us_arima_rmse[best_us_idx], "\n\n")

# ============================================================================
# 8. In-sample fit metrics
# ============================================================================
fit_can_arima <- Arima(train_can$infl, order = best_can_order)
fit_us_arima <- Arima(train_us$infl, order = best_us_order)

can_train_rmse <- rmse(train_can$infl, fitted(fit_can_arima))
us_train_rmse <- rmse(train_us$infl, fitted(fit_us_arima))

# ============================================================================
# 9. Compile results
# ============================================================================
arima_holdout_results <- tibble(
  Country = c("Canada", "United States"),
  Model = "ARIMA",
  Order = c(paste(best_can_order, collapse = ","), paste(best_us_order, collapse = ",")),
  Holdout_RMSE = c(can_arima_rmse[best_can_idx], us_arima_rmse[best_us_idx]),
  Training_RMSE = c(can_train_rmse, us_train_rmse)
)

cat("\n=== ARIMA FINAL RESULTS ===\n")
print(arima_holdout_results)

# ============================================================================
# 10. Candidate comparison tables (for appendix)
# ============================================================================
arima_candidates_table <- tibble(
  Country = c(rep("Canada", length(arima_candidates)), rep("United States", length(arima_candidates))),
  ARIMA_Order = rep(sapply(arima_candidates, function(o) paste("(", paste(o, collapse = ","), ")", sep = "")), 2),
  Holdout_RMSE = c(can_arima_rmse, us_arima_rmse)
) %>% arrange(Country, Holdout_RMSE)

# ============================================================================
# 11. Save results
# ============================================================================
arima_results_path <- file.path(output_tables_dir, "arima_final_results.csv")
readr::write_csv(arima_holdout_results, arima_results_path)

arima_candidates_path <- file.path(output_tables_dir, "arima_candidates_comparison.csv")
readr::write_csv(arima_candidates_table, arima_candidates_path)

# Save predictions
arima_pred_df <- tibble(
  country = c(rep("Canada", nrow(test_can)), rep("United States", nrow(test_us))),
  month = c(test_can$month, test_us$month),
  actual = c(test_can$infl, test_us$infl),
  pred_arima = c(best_can_preds, best_us_preds)
)

arima_pred_path <- file.path(output_tables_dir, "arima_predictions.csv")
readr::write_csv(arima_pred_df, arima_pred_path)

# ============================================================================
# 12. Visualizations
# ============================================================================
# Plot: Actual vs ARIMA forecasts
arima_plot_data <- arima_pred_df %>%
  pivot_longer(cols = c(actual, pred_arima), names_to = "type", values_to = "value") %>%
  mutate(type = recode(type, actual = "Actual", pred_arima = "ARIMA Forecast"))

p_arima <- ggplot(arima_plot_data, aes(x = month, y = value, color = type, linetype = type)) +
  geom_line(size = 0.5) +
  scale_color_manual(values = c("Actual" = "black", "ARIMA Forecast" = "firebrick")) +
  scale_linetype_manual(values = c("Actual" = "solid", "ARIMA Forecast" = "dashed")) +
  facet_wrap(~ country, scales = "free_y", ncol = 1) +
  labs(
    title = "ARIMA: Actual vs Holdout Forecasts",
    x = NULL,
    y = "Inflation (log difference)",
    color = NULL,
    linetype = NULL
  ) +
  theme_minimal() +
  theme(legend.position = "bottom")

arima_plot_path <- file.path(output_figures_dir, "arima_holdout_forecasts.png")
ggsave(arima_plot_path, p_arima, width = 9, height = 6, dpi = 150)

# ============================================================================
# 13. Model diagnostics
# ============================================================================
png(
  file.path(output_figures_dir, "arima_diagnostics.png"),
  width = 1000, height = 600
)
par(mfrow = c(1, 2), mar = c(4, 4, 3, 1))

acf(residuals(fit_can_arima), main = paste("Canada ARIMA(", paste(best_can_order, collapse = ","), ") Residuals"), lag.max = 20)
acf(residuals(fit_us_arima), main = paste("US ARIMA(", paste(best_us_order, collapse = ","), ") Residuals"), lag.max = 20)

dev.off()

cat("\n ARIMA analysis complete!")
cat("\nFiles saved:")
cat("\n  - ", arima_results_path)
cat("\n  - ", arima_candidates_path)
cat("\n  - ", arima_pred_path)
cat("\n  - ", arima_plot_path, "\n")

# Save workspace
save(
  best_can_order, best_us_order,
  best_can_preds, best_us_preds,
  fit_can_arima, fit_us_arima,
  arima_holdout_results,
  file = file.path(data_processed_dir, "arima_results.RData")
)
