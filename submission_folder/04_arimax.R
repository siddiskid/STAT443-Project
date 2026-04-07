# ============================================================================
# 04_arimax.R
# ARIMAX models with exogenous variables for Canada and US inflation
# Tests different variable combinations and uses best for holdout forecasting
# Output: Results and diagnostics saved to output/tables and output/figures
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
load(file.path(data_processed_dir, "arima_results.RData"))

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

# ARIMAX 1-step ahead expanding window forecast
arimax_expanding_forecast <- function(train_y, test_y, train_x, test_x, 
                                       order = NULL, series_name = "") {
  cat("ARIMAX for", series_name, "...\n")
  
  preds <- numeric(length(test_y$infl))
  y <- train_y$infl
  x <- as.matrix(train_x)
  
  for (i in seq_len(nrow(test_y))) {
    fit <- if (is.null(order)) {
      auto.arima(y, xreg = x, seasonal = FALSE, stepwise = TRUE, approximation = FALSE)
    } else {
      Arima(y, order = order, xreg = x)
    }
    
    x_next <- as.matrix(test_x[i, , drop = FALSE])
    pr <- predict(fit, n.ahead = 1, newxreg = x_next)
    preds[i] <- as.numeric(pr$pred[1])
    
    y <- c(y, test_y$infl[i])
    x <- rbind(x, x_next)
  }
  
  preds
}

# ============================================================================
# 4. Cross-correlation analysis (diagnostics)
# ============================================================================
cat("\n=== CROSS-CORRELATION ANALYSIS ===\n\n")

png(
  file.path(output_figures_dir, "arimax_ccf_diagnostics.png"),
  width = 1500,     
  height = 1000,    
  res = 150,      
  pointsize = 10   
)

par(
  mfrow = c(2, 3), 
  mar = c(5, 5, 4, 2), 
  oma = c(1, 1, 5, 1), 
  cex.main = 1.1, 
  cex.lab = 1.0
)
# Canada CCF
ccf(train_can$infl, train_can$oil_wti_l1, main = "Canada: infl vs oil", lag.max = 12)
ccf(train_can$infl, train_can$ippi_can_l1, main = "Canada: infl vs IPPI", lag.max = 12)
ccf(train_can$infl, train_can$gdp_can_l1, main = "Canada: infl vs GDP", lag.max = 12)

# US CCF
ccf(train_us$infl, train_us$oil_wti_l1, main = "US: infl vs oil", lag.max = 12)
ccf(train_us$infl, train_us$ppi_us_l1, main = "US: infl vs PPI", lag.max = 12)
ccf(train_us$infl, train_us$mxp_us_l1, main = "US: infl vs MXP", lag.max = 12)

mtext("Cross-Correlation Function (CCF) Diagnostics", outer = TRUE, cex = 1.4, font = 2, line = 1.5)

dev.off()
cat("Saved CCF diagnostics plot\n")

# ============================================================================
# 5. ARIMAX specifications for Canada
# ============================================================================
cat("\n=== TESTING ARIMAX SPECIFICATIONS FOR CANADA ===\n\n")

can_arimax_specs <- list(
  oil = list(
    train = train_can[, c("oil_wti_l1"), drop = FALSE],
    test = test_can[, c("oil_wti_l1"), drop = FALSE],
    description = "Oil"
  ),
  oil_ippi = list(
    train = train_can[, c("oil_wti_l1", "ippi_can_l1")],
    test = test_can[, c("oil_wti_l1", "ippi_can_l1")],
    description = "Oil + IPPI"
  )
)

can_arimax_rmse <- numeric(length(can_arimax_specs))
names(can_arimax_rmse) <- sapply(can_arimax_specs, function(s) s$description)

for (i in seq_along(can_arimax_specs)) {
  spec <- can_arimax_specs[[i]]
  preds <- arimax_expanding_forecast(
    train_can, test_can,
    spec$train, spec$test,
    series_name = paste("Canada -", spec$description)
  )
  can_arimax_rmse[i] <- rmse(test_can$infl, preds)
  cat("  RMSE:", can_arimax_rmse[i], "\n")
}

best_can_arimax_idx <- which.min(can_arimax_rmse)
best_can_arimax_spec <- can_arimax_specs[[best_can_arimax_idx]]
best_can_arimax_preds <- arimax_expanding_forecast(
  train_can, test_can,
  best_can_arimax_spec$train, best_can_arimax_spec$test,
  series_name = paste("Canada - Best ARIMAX:", best_can_arimax_spec$description)
)

cat("\nCanada ARIMAX Results:\n")
print(tibble(Specification = names(can_arimax_rmse), RMSE = can_arimax_rmse) %>% arrange(RMSE))
cat("Best specification for Canada:", best_can_arimax_spec$description, "\n")
cat("Holdout RMSE:", can_arimax_rmse[best_can_arimax_idx], "\n\n")

# ============================================================================
# 6. ARIMAX specifications for US
# ============================================================================
cat("=== TESTING ARIMAX SPECIFICATIONS FOR US ===\n\n")

us_arimax_specs <- list(
  oil_ppi = list(
    train = train_us[, c("oil_wti_l1", "ppi_us_l1")],
    test = test_us[, c("oil_wti_l1", "ppi_us_l1")],
    description = "Oil + PPI"
  ),
  oil_ppi_mxp = list(
    train = train_us[, c("oil_wti_l1", "ppi_us_l1", "mxp_us_l1")],
    test = test_us[, c("oil_wti_l1", "ppi_us_l1", "mxp_us_l1")],
    description = "Oil + PPI + MXP"
  ),
  oil_ppi_mxp_u6 = list(
    train = train_us[, c("oil_wti_l1", "ppi_us_l1", "mxp_us_l1", "u6_us_l1")],
    test = test_us[, c("oil_wti_l1", "ppi_us_l1", "mxp_us_l1", "u6_us_l1")],
    description = "Oil + PPI + MXP + U6"
  )
)

us_arimax_rmse <- numeric(length(us_arimax_specs))
names(us_arimax_rmse) <- sapply(us_arimax_specs, function(s) s$description)

for (i in seq_along(us_arimax_specs)) {
  spec <- us_arimax_specs[[i]]
  preds <- arimax_expanding_forecast(
    train_us, test_us,
    spec$train, spec$test,
    series_name = paste("US -", spec$description)
  )
  us_arimax_rmse[i] <- rmse(test_us$infl, preds)
  cat("  RMSE:", us_arimax_rmse[i], "\n")
}

best_us_arimax_idx <- which.min(us_arimax_rmse)
best_us_arimax_spec <- us_arimax_specs[[best_us_arimax_idx]]
best_us_arimax_preds <- arimax_expanding_forecast(
  train_us, test_us,
  best_us_arimax_spec$train, best_us_arimax_spec$test,
  series_name = paste("US - Best ARIMAX:", best_us_arimax_spec$description)
)

cat("\nUS ARIMAX Results:\n")
print(tibble(Specification = names(us_arimax_rmse), RMSE = us_arimax_rmse) %>% arrange(RMSE))
cat("Best specification for US:", best_us_arimax_spec$description, "\n")
cat("Holdout RMSE:", us_arimax_rmse[best_us_arimax_idx], "\n\n")

# ============================================================================
# 7. In-sample fit
# ============================================================================
fit_can_arimax <- auto.arima(
  train_can$infl,
  xreg = as.matrix(best_can_arimax_spec$train),
  seasonal = FALSE, stepwise = TRUE, approximation = FALSE
)
fit_us_arimax <- auto.arima(
  train_us$infl,
  xreg = as.matrix(best_us_arimax_spec$train),
  seasonal = FALSE, stepwise = TRUE, approximation = FALSE
)

can_arimax_train_rmse <- rmse(train_can$infl, fitted(fit_can_arimax))
us_arimax_train_rmse <- rmse(train_us$infl, fitted(fit_us_arimax))

# ============================================================================
# 8. Compile results
# ============================================================================
arimax_holdout_results <- tibble(
  Country = c("Canada", "United States"),
  Model = "ARIMAX",
  Variables = c(best_can_arimax_spec$description, best_us_arimax_spec$description),
  Holdout_RMSE = c(can_arimax_rmse[best_can_arimax_idx], us_arimax_rmse[best_us_arimax_idx]),
  Training_RMSE = c(can_arimax_train_rmse, us_arimax_train_rmse)
)

cat("\n=== ARIMAX FINAL RESULTS ===\n")
print(arimax_holdout_results)

# ============================================================================
# 9. Candidate specifications table (for appendix)
# ============================================================================
arimax_candidates_table <- tibble(
  Country = c(rep("Canada", length(can_arimax_specs)), rep("United States", length(us_arimax_specs))),
  Specification = c(sapply(can_arimax_specs, function(s) s$description),
                    sapply(us_arimax_specs, function(s) s$description)),
  Holdout_RMSE = c(can_arimax_rmse, us_arimax_rmse)
) %>% arrange(Country, Holdout_RMSE)

# ============================================================================
# 10. Save results
# ============================================================================
arimax_results_path <- file.path(output_tables_dir, "arimax_final_results.csv")
readr::write_csv(arimax_holdout_results, arimax_results_path)

arimax_candidates_path <- file.path(output_tables_dir, "arimax_candidates_comparison.csv")
readr::write_csv(arimax_candidates_table, arimax_candidates_path)

# Save predictions
arimax_pred_df <- tibble(
  country = c(rep("Canada", nrow(test_can)), rep("United States", nrow(test_us))),
  month = c(test_can$month, test_us$month),
  actual = c(test_can$infl, test_us$infl),
  pred_arimax = c(best_can_arimax_preds, best_us_arimax_preds)
)

arimax_pred_path <- file.path(output_tables_dir, "arimax_predictions.csv")
readr::write_csv(arimax_pred_df, arimax_pred_path)

# ============================================================================
# 11. Visualizations
# ============================================================================
arimax_plot_data <- arimax_pred_df %>%
  pivot_longer(cols = c(actual, pred_arimax), names_to = "type", values_to = "value") %>%
  mutate(type = recode(type, actual = "Actual", pred_arimax = "ARIMAX Forecast"))

p_arimax <- ggplot(arimax_plot_data, aes(x = month, y = value, color = type, linetype = type)) +
  geom_line(size = 0.5) +
  scale_color_manual(values = c("Actual" = "black", "ARIMAX Forecast" = "darkgreen")) +
  scale_linetype_manual(values = c("Actual" = "solid", "ARIMAX Forecast" = "dashed")) +
  facet_wrap(~ country, scales = "free_y", ncol = 1) +
  labs(
    title = "ARIMAX: Actual vs Holdout Forecasts",
    x = NULL,
    y = "Inflation (log difference)",
    color = NULL,
    linetype = NULL
  ) +
  theme_minimal() +
  theme(legend.position = "bottom")

arimax_plot_path <- file.path(output_figures_dir, "arimax_holdout_forecasts.png")
ggsave(arimax_plot_path, p_arimax, width = 9, height = 6, dpi = 150)

cat("\n ARIMAX analysis complete!")
cat("\nFiles saved:")
cat("\n  - ", arimax_results_path)
cat("\n  - ", arimax_candidates_path)
cat("\n  - ", arimax_pred_path)
cat("\n  - ", arimax_plot_path, "\n")

# Save workspace
save(
  best_can_arimax_spec, best_us_arimax_spec,
  best_can_arimax_preds, best_us_arimax_preds,
  fit_can_arimax, fit_us_arimax,
  arimax_holdout_results,
  file = file.path(data_processed_dir, "arimax_results.RData")
)
