# ============================================================================
# 05_forecast_comparison.R
# Compare all forecasting models: simple rules, ETS, ARIMA, ARIMAX
# Generates final summary tables and ranking plots
# Output: Comprehensive comparison tables and visualizations
# ============================================================================

suppressPackageStartupMessages({
  library(tidyverse)
  library(forecast)
})

options(dplyr.summarise.inform = FALSE)

# ============================================================================
# 1. Setup and load all results
# ============================================================================
data_processed_dir <- file.path(getwd(), "data_processed")
output_tables_dir <- file.path(getwd(), "output", "tables")
output_figures_dir <- file.path(getwd(), "output", "figures")

dir.create(output_tables_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(output_figures_dir, recursive = TRUE, showWarnings = FALSE)

load(file.path(data_processed_dir, "wrangled_data.RData"))
load(file.path(data_processed_dir, "ets_results.RData"))
load(file.path(data_processed_dir, "arima_results.RData"))
load(file.path(data_processed_dir, "arimax_results.RData"))

# ============================================================================
# 2. Data split (same as in previous scripts)
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

maex <- function(actual, pred) {
  mean(abs(actual - pred), na.rm = TRUE)
}

# ============================================================================
# 4. Simple rules: Persistence and Mean
# ============================================================================
cat("\n=== SIMPLE BASELINE RULES ===\n\n")

# Persistence: use last value of training set and repeat
persistence_expanding <- function(train, test) {
  preds <- numeric(nrow(test))
  last <- tail(train$infl, 1)
  for (i in seq_len(nrow(test))) {
    preds[i] <- last
    last <- test$infl[i]
  }
  preds
}

# Mean rule: use training set mean
mean_rule <- function(train, test) {
  rep(mean(train$infl, na.rm = TRUE), nrow(test))
}

# In-sample metrics for comparison
mean_insample <- function(train) {
  rep(mean(train$infl, na.rm = TRUE), nrow(train))
}

persistence_insample <- function(train) {
  c(NA, train$infl[-nrow(train)])
}

# Canada simple rules
persist_can_preds <- persistence_expanding(train_can, test_can)
mean_can_preds <- mean_rule(train_can, test_can)

persist_can_train <- persistence_insample(train_can)
mean_can_train <- mean_insample(train_can)

# US simple rules
persist_us_preds <- persistence_expanding(train_us, test_us)
mean_us_preds <- mean_rule(train_us, test_us)

persist_us_train <- persistence_insample(train_us)
mean_us_train <- mean_insample(train_us)

# Calculate RMSE
persist_can_rmse_test <- rmse(test_can$infl, persist_can_preds)
persist_can_rmse_train <- rmse(train_can$infl[-1], persist_can_train[-1])

mean_can_rmse_test <- rmse(test_can$infl, mean_can_preds)
mean_can_rmse_train <- rmse(train_can$infl, mean_can_train)

persist_us_rmse_test <- rmse(test_us$infl, persist_us_preds)
persist_us_rmse_train <- rmse(train_us$infl[-1], persist_us_train[-1])

mean_us_rmse_test <- rmse(test_us$infl, mean_us_preds)
mean_us_rmse_train <- rmse(train_us$infl, mean_us_train)

simple_rules_results <- tibble(
  Country = c("Canada", "Canada", "United States", "United States"),
  Model = c("Persistence", "Mean", "Persistence", "Mean"),
  Training_RMSE = c(persist_can_rmse_train, mean_can_rmse_train, persist_us_rmse_train, mean_us_rmse_train),
  Holdout_RMSE = c(persist_can_rmse_test, mean_can_rmse_test, persist_us_rmse_test, mean_us_rmse_test)
)

cat("Simple Rules Results:\n")
print(simple_rules_results)

# ============================================================================
# 5. Compile all results
# ============================================================================
cat("\n=== OVERALL MODEL COMPARISON ===\n\n")

# Read individual results
ets_results_df <- readr::read_csv(
  file.path(output_tables_dir, "ets_results.csv"),
  show_col_types = FALSE
) %>% select(Country, Model, Training_RMSE, Holdout_RMSE) %>%
  filter(!is.na(Training_RMSE))

arima_results_df <- readr::read_csv(
  file.path(output_tables_dir, "arima_final_results.csv"),
  show_col_types = FALSE
) %>% select(Country, Model, Training_RMSE, Holdout_RMSE)

arimax_results_df <- readr::read_csv(
  file.path(output_tables_dir, "arimax_final_results.csv"),
  show_col_types = FALSE
) %>% select(Country, Model, Training_RMSE, Holdout_RMSE)

# Combine all results
all_models_comparison <- bind_rows(
  simple_rules_results,
  ets_results_df,
  arima_results_df,
  arimax_results_df
) %>%
  arrange(Country, Holdout_RMSE)

cat("All Models Ranked by Holdout RMSE:\n\n")
print(all_models_comparison)

# ============================================================================
# 6. Summary: Best model by country
# ============================================================================
best_by_country <- all_models_comparison %>%
  group_by(Country) %>%
  slice(1) %>%
  ungroup()

cat("\n=== BEST MODEL BY COUNTRY ===\n\n")
print(best_by_country)

# ============================================================================
# 7. Save comprehensive results
# ============================================================================
comparison_path <- file.path(output_tables_dir, "model_comparison_all.csv")
readr::write_csv(all_models_comparison, comparison_path)

best_path <- file.path(output_tables_dir, "best_model_by_country.csv")
readr::write_csv(best_by_country, best_path)

# ============================================================================
# 8. Visualizations: RMSE Comparison
# ============================================================================
p_rmse <- all_models_comparison %>%
  ggplot(aes(x = reorder(Model, Holdout_RMSE), y = Holdout_RMSE, fill = Country)) +
  geom_col(position = position_dodge(width = 0.8), width = 0.7) +
  coord_flip() +
  labs(
    title = "Holdout RMSE by Model and Country",
    x = "Model",
    y = "RMSE (log difference scale)",
    fill = "Country"
  ) +
  theme_minimal() +
  theme(legend.position = "bottom")

rmse_plot_path <- file.path(output_figures_dir, "rmse_comparison_all_models.png")
ggsave(rmse_plot_path, p_rmse, width = 9, height = 6, dpi = 150)

# ============================================================================
# 9. Visualizations: Training vs Holdout
# ============================================================================
p_train_test <- all_models_comparison %>%
  pivot_longer(cols = c(Training_RMSE, Holdout_RMSE), names_to = "Set", values_to = "RMSE") %>%
  mutate(Set = recode(Set, Training_RMSE = "Training", Holdout_RMSE = "Holdout")) %>%
  ggplot(aes(x = reorder(Model, RMSE), y = RMSE, fill = Set)) +
  geom_col(position = position_dodge(width = 0.8), width = 0.7) +
  coord_flip() +
  facet_wrap(~ Country, scales = "free_x") +
  labs(
    title = "Training vs Holdout RMSE by Model",
    x = "Model",
    y = "RMSE",
    fill = "Set"
  ) +
  theme_minimal() +
  theme(legend.position = "bottom")

train_test_plot_path <- file.path(output_figures_dir, "training_vs_holdout_rmse.png")
ggsave(train_test_plot_path, p_train_test, width = 10, height = 6, dpi = 150)

# ============================================================================
# 10. Load all predictions and combine
# ============================================================================
simple_predictions <- tibble(
  country = c(rep("Canada", nrow(test_can)), rep("United States", nrow(test_us))),
  month = c(test_can$month, test_us$month),
  actual = c(test_can$infl, test_us$infl),
  pred_persistence = c(persist_can_preds, persist_us_preds),
  pred_mean = c(mean_can_preds, mean_us_preds)
)

ets_pred <- readr::read_csv(
  file.path(output_tables_dir, "ets_predictions.csv"),
  show_col_types = FALSE
) %>%
  rename(pred_ets = pred_ets)

arima_pred <- readr::read_csv(
  file.path(output_tables_dir, "arima_predictions.csv"),
  show_col_types = FALSE
) %>%
  rename(pred_arima = pred_arima)

arimax_pred <- readr::read_csv(
  file.path(output_tables_dir, "arimax_predictions.csv"),
  show_col_types = FALSE
) %>%
  rename(pred_arimax = pred_arimax)

# Combine all predictions
all_predictions <- simple_predictions %>%
  left_join(ets_pred %>% select(country, month, pred_ets), by = c("country", "month")) %>%
  left_join(arima_pred %>% select(country, month, pred_arima), by = c("country", "month")) %>%
  left_join(arimax_pred %>% select(country, month, pred_arimax), by = c("country", "month"))

all_pred_path <- file.path(output_tables_dir, "all_predictions_combined.csv")
readr::write_csv(all_predictions, all_pred_path)

# ============================================================================
# 11. Visualization: Time series comparison for best models
# ============================================================================
best_can_model <- best_by_country %>% filter(Country == "Canada") %>% pull(Model)
best_us_model <- best_by_country %>% filter(Country == "United States") %>% pull(Model)

# Prepare data for best models
best_pred_cols <- c("actual", tolower(paste0("pred_", gsub(" |\\(|\\)", "_", tolower(best_can_model)))))

best_predictions <- all_predictions %>%
  pivot_longer(
    cols = starts_with("pred_"),
    names_to = "model",
    values_to = "pred"
  ) %>%
  mutate(model = str_remove(model, "pred_")) %>%
  filter(
    (country == "Canada" & model == tolower(gsub(" |\\(|\\)", "_", best_can_model))) |
    (country == "United States" & model == tolower(gsub(" |\\(|\\)", "_", best_us_model))) |
    TRUE  # Keep actual
  )

# ============================================================================
# 12. Summary for report
# ============================================================================
cat("\n✓ Forecast comparison complete!")
cat("\nKey Summary Statistics:\n")
cat("\nCanada:\n")
can_summary <- all_models_comparison %>% filter(Country == "Canada")
print(can_summary)

cat("\nUnited States:\n")
us_summary <- all_models_comparison %>% filter(Country == "United States")
print(us_summary)

cat("\nBest models:\n")
print(best_by_country)

cat("\n Files saved:")
cat("\n  - ", comparison_path)
cat("\n  - ", best_path)
cat("\n  - ", rmse_plot_path)
cat("\n  - ", train_test_plot_path)
cat("\n  - ", all_pred_path, "\n")

# Final save
save(
  all_models_comparison,
  best_by_country,
  simple_rules_results,
  all_predictions,
  file = file.path(data_processed_dir, "forecast_comparison_final.RData")
)
