# ============================================================================
# 02_exponential_smoothing.R
# Exponential Smoothing (ETS) models for Canada and US inflation
# Fit ETS models on training set, evaluate on holdout set
# Output: Tables and plots saved to output/tables and output/figures
# ============================================================================

suppressPackageStartupMessages({
  library(tidyverse)
  library(forecast)
})

options(dplyr.summarise.inform = FALSE)

# ============================================================================
# 1. Setup directories and load data
# ============================================================================
data_processed_dir <- file.path(getwd(), "data_processed")
output_tables_dir <- file.path(getwd(), "output", "tables")
output_figures_dir <- file.path(getwd(), "output", "figures")

dir.create(output_tables_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(output_figures_dir, recursive = TRUE, showWarnings = FALSE)

load(file.path(data_processed_dir, "wrangled_data.RData"))

# ============================================================================
# 2. Set up training and holdout split
# ============================================================================
h <- 24  # Holdout period = 24 months

# Canada
n_can <- nrow(canada_common)
train_can <- canada_common[1:(n_can - h), ]
test_can  <- canada_common[(n_can - h + 1):n_can, ]

# US
n_us <- nrow(us_common)
train_us <- us_common[1:(n_us - h), ]
test_us  <- us_common[(n_us - h + 1):n_us, ]

cat("Data split summary:\n")
cat("Canada: training=", nrow(train_can), " test=", nrow(test_can), "\n")
cat("US:     training=", nrow(train_us), "   test=", nrow(test_us), "\n\n")

# ============================================================================
# 3. Helper function: RMSE
# ============================================================================
rmse <- function(actual, pred) {
  sqrt(mean((actual - pred)^2, na.rm = TRUE))
}

# ============================================================================
# 4. ETS: 1-step ahead expanding window forecast
# ============================================================================
ets_expanding_forecast <- function(train, test, series_name) {
  cat("ETS expanding window for", series_name, "...\n")
  
  preds <- numeric(nrow(test))
  y <- train$infl
  
  for (i in seq_len(nrow(test))) {
    fit <- ets(y, model = "ZZZ", opt.crit = "lik")
    preds[i] <- as.numeric(forecast(fit, h = 1)$mean[1])
    y <- c(y, test$infl[i])
  }
  
  preds
}

# ============================================================================
# 5. Fit ETS models and get forecasts
# ============================================================================
cat("\n=== FITTING ETS MODELS ===\n\n")

ets_preds_can <- ets_expanding_forecast(train_can, test_can, "Canada")
ets_preds_us <- ets_expanding_forecast(train_us, test_us, "United States")

# ============================================================================
# 6. Calculate holdout RMSE for ETS
# ============================================================================
ets_rmse_can <- rmse(test_can$infl, ets_preds_can)
ets_rmse_us <- rmse(test_us$infl, ets_preds_us)

ets_results <- tibble(
  Country = c("Canada", "United States"),
  Model = "Exponential Smoothing (ETS)",
  Holdout_RMSE = c(ets_rmse_can, ets_rmse_us)
)

cat("=== ETS Holdout RMSE ===\n")
print(ets_results)

# ============================================================================
# 7. Also compare in-sample fitted values for completeness
# ============================================================================
fit_ets_can <- ets(train_can$infl, model = "ZZZ", opt.crit = "lik")
fit_ets_us <- ets(train_us$infl, model = "ZZZ", opt.crit = "lik")

train_rmse_ets_can <- rmse(train_can$infl, fitted(fit_ets_can))
train_rmse_ets_us <- rmse(train_us$infl, fitted(fit_ets_us))

ets_train_results <- tibble(
  Country = c("Canada", "United States"),
  Model = "Exponential Smoothing (ETS)",
  Training_RMSE = c(train_rmse_ets_can, train_rmse_ets_us)
)

cat("\n=== ETS Training RMSE ===\n")
print(ets_train_results)

# ============================================================================
# 8. Save results
# ============================================================================
ets_all_results <- ets_results %>%
  left_join(ets_train_results, by = c("Country", "Model"))

ets_path <- file.path(output_tables_dir, "ets_results.csv")
readr::write_csv(ets_all_results, ets_path)

# Save predictions for later comparison
ets_pred_df <- tibble(
  country = c(rep("Canada", nrow(test_can)), rep("United States", nrow(test_us))),
  month = c(test_can$month, test_us$month),
  actual = c(test_can$infl, test_us$infl),
  pred_ets = c(ets_preds_can, ets_preds_us)
)

ets_pred_path <- file.path(output_tables_dir, "ets_predictions.csv")
readr::write_csv(ets_pred_df, ets_pred_path)

# ============================================================================
# 9. Visualize ETS forecasts
# ============================================================================
ets_plot_data <- ets_pred_df %>%
  pivot_longer(cols = c(actual, pred_ets), names_to = "type", values_to = "value") %>%
  mutate(type = recode(type, actual = "Actual", pred_ets = "ETS Forecast"))

p_ets <- ggplot(ets_plot_data, aes(x = month, y = value, color = type, linetype = type)) +
  geom_line(size = 0.5) +
  scale_color_manual(values = c("Actual" = "black", "ETS Forecast" = "steelblue")) +
  scale_linetype_manual(values = c("Actual" = "solid", "ETS Forecast" = "dashed")) +
  facet_wrap(~ country, scales = "free_y", ncol = 1) +
  labs(
    title = "Exponential Smoothing: Actual vs Holdout Forecasts",
    x = NULL,
    y = "Inflation (log difference)",
    color = NULL,
    linetype = NULL
  ) +
  theme_minimal() +
  theme(legend.position = "bottom")

ets_plot_path <- file.path(output_figures_dir, "ets_holdout_forecasts.png")
ggsave(ets_plot_path, p_ets, width = 9, height = 6, dpi = 150)

cat("\n ETS analysis complete!")
cat("\nFiles saved:")
cat("\n  - ", ets_path)
cat("\n  - ", ets_pred_path)
cat("\n  - ", ets_plot_path, "\n")

# Save workspace for next stage
save(
  ets_preds_can, ets_preds_us, ets_rmse_can, ets_rmse_us,
  ets_results, ets_all_results,
  fit_ets_can, fit_ets_us,
  file = file.path(data_processed_dir, "ets_results.RData")
)
