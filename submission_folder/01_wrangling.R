# ============================================================================
# 01_wrangling.R
# Data wrangling and preparation for STAT443 Project
# Loads raw data, performs transformations, creates training/test splits
# Output: data_processed/canada_model_common_window.csv, us_model_common_window.csv
# ============================================================================

suppressPackageStartupMessages({
  library(tidyverse)
  library(lubridate)
  library(readxl)
})

options(dplyr.summarise.inform = FALSE)

# ============================================================================
# 1. Setup and file paths
# ============================================================================
proj_dir <- getwd()
data_dir <- file.path(proj_dir, "data")
data_clean_dir <- file.path(proj_dir, "data_clean")
data_processed_dir <- file.path(proj_dir, "data_processed")

# Create directories if they don't exist
dir.create(data_clean_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(data_processed_dir, recursive = TRUE, showWarnings = FALSE)

# ============================================================================
# 2. Helper functions for date parsing and reading
# ============================================================================
parse_month_date <- function(x) {
  x <- as.character(x)
  out <- suppressWarnings(lubridate::parse_date_time(
    x,
    orders = c("ymd", "Ymd", "b-y", "b Y", "Y b", "Y-m-d", "Y/m/d")
  ))
  as.Date(out)
}

read_series <- function(series_id, file_name, date_col, value_col, file_type = "csv") {
  path <- file.path(data_dir, file_name)
  cat("Reading:", file_name, "\n")
  
  if (file_type == "xlsx") {
    df <- readxl::read_excel(path)
  } else {
    df <- readr::read_csv(path, show_col_types = FALSE)
  }

  tibble(
    series_id = series_id,
    date_raw = df[[date_col]],
    value = suppressWarnings(as.numeric(df[[value_col]]))
  ) %>%
    mutate(date = parse_month_date(date_raw)) %>%
    select(series_id, date, value, date_raw) %>%
    arrange(date)
}

# ============================================================================
# 3. Load all data series
# ============================================================================
series_data <- list(
  cpi_can = read_series("cpi_can", "cpi_canada_t.csv", "REF_DATE", "VALUE"),
  cpi_us = read_series("cpi_us", "cpi_usa.csv", "observation_date", "CPIAUCSL"),
  oil_wti = read_series("oil_wti", "crude_oil.csv", "observation_date", "MCOILWTICO"),
  u6_us = read_series("u6_us", "labour_usa.csv", "observation_date", "U6RATE"),
  ppi_us = read_series("ppi_us", "ppi_usa (2).csv", "observation_date", "WPSFD49208"),
  mxp_us = read_series("mxp_us", "mxp_usa.csv", "Label", "Value"),
  exrate_can = read_series("exrate_can", "can_exchange_rate.csv", "TIME_PERIOD:Period", "OBS_VALUE:Value"),
  gdp_can = read_series("gdp_can", "real_gdp_canada_t.csv", "REF_DATE", "VALUE"),
  gdp_us = read_series("gdp_us", "real_gdp_usa.csv", "observation_date", "BBKMGDP"),
  ippi_can = read_series("ippi_can", "ippi_can.xlsx", "REF_DATE", "VALUE", file_type = "xlsx"),
  unemp_can = read_series("unemp_can", "labour_canada_t.xlsx", "REF_DATE", "VALUE", file_type = "xlsx")
)

# ============================================================================
# 4. Stack all series and save
# ============================================================================
stacked_df <- bind_rows(purrr::map(series_data, ~ mutate(.x, date_raw = as.character(date_raw))))
stacked_path <- file.path(data_clean_dir, "series_stacked.csv")
readr::write_csv(stacked_df, stacked_path)
cat("Saved stacked series:", stacked_path, "\n")

# ============================================================================
# 5. Create wide panel and transformations
# ============================================================================
panel_wide <- bind_rows(
  purrr::map(series_data, ~ .x %>% 
    mutate(month = as.Date(lubridate::floor_date(date, "month"))) %>%
    select(series_id, month, value))
) %>%
  group_by(series_id, month) %>%
  summarise(value = dplyr::first(value), .groups = "drop") %>%
  tidyr::pivot_wider(names_from = series_id, values_from = value) %>%
  arrange(month)

# Apply transformations: inflation targets and lagged predictors
panel_transformed <- panel_wide %>%
  mutate(
    # Inflation targets (log differences)
    infl_can = log(cpi_can) - log(lag(cpi_can)),
    infl_us = log(cpi_us) - log(lag(cpi_us)),
    
    # Log-difference predictors
    oil_wti_ld = log(oil_wti) - log(lag(oil_wti)),
    ppi_us_ld = log(ppi_us) - log(lag(ppi_us)),
    mxp_us_ld = log(mxp_us) - log(lag(mxp_us)),
    exrate_can_ld = log(exrate_can) - log(lag(exrate_can)),
    ippi_can_ld = log(ippi_can) - log(lag(ippi_can)),
    gdp_can_ld = log(gdp_can) - log(lag(gdp_can)),
    
    # Keep as-is
    gdp_us_rate = gdp_us,
    u6_us_lvl = u6_us,
    unemp_can_lvl = unemp_can
  ) %>%
  # Create lag-1 predictors
  mutate(
    oil_wti_l1 = lag(oil_wti_ld),
    ppi_us_l1 = lag(ppi_us_ld),
    mxp_us_l1 = lag(mxp_us_ld),
    exrate_can_l1 = lag(exrate_can_ld),
    ippi_can_l1 = lag(ippi_can_ld),
    gdp_can_l1 = lag(gdp_can_ld),
    gdp_us_l1 = lag(gdp_us_rate),
    u6_us_l1 = lag(u6_us_lvl),
    unemp_can_l1 = lag(unemp_can_lvl)
  )

# ============================================================================
# 6. Create model datasets: Canada and US
# ============================================================================
canada_model_raw <- panel_transformed %>%
  transmute(
    month,
    infl = infl_can,
    gdp_can_l1,
    unemp_can_l1,
    ippi_can_l1,
    oil_wti_l1,
    exrate_can_l1
  )

us_model_raw <- panel_transformed %>%
  transmute(
    month,
    infl = infl_us,
    gdp_us_l1,
    u6_us_l1,
    ppi_us_l1,
    oil_wti_l1,
    mxp_us_l1
  )

# Remove rows with missing values
canada_model <- canada_model_raw %>% filter(if_all(-month, ~ !is.na(.)))
us_model <- us_model_raw %>% filter(if_all(-month, ~ !is.na(.)))

# Find common time window
common_start <- max(min(canada_model$month), min(us_model$month))
common_end <- min(max(canada_model$month), max(us_model$month))

canada_common <- canada_model %>% filter(month >= common_start, month <= common_end)
us_common <- us_model %>% filter(month >= common_start, month <= common_end)

# ============================================================================
# 7. Save processed datasets
# ============================================================================
canada_path <- file.path(data_processed_dir, "canada_model_full_sample.csv")
us_path <- file.path(data_processed_dir, "us_model_full_sample.csv")
canada_common_path <- file.path(data_processed_dir, "canada_model_common_window.csv")
us_common_path <- file.path(data_processed_dir, "us_model_common_window.csv")

readr::write_csv(canada_model, canada_path)
readr::write_csv(us_model, us_path)
readr::write_csv(canada_common, canada_common_path)
readr::write_csv(us_common, us_common_path)

# ============================================================================
# 8. Summary and diagnostics
# ============================================================================
window_summary <- tibble::tibble(
  dataset = c("Canada (full)", "US (full)", "Common window"),
  start_date = as.character(c(
    min(canada_model$month), 
    min(us_model$month), 
    common_start
  )),
  end_date = as.character(c(
    max(canada_model$month),
    max(us_model$month),
    common_end
  )),
  n_obs = c(nrow(canada_model), nrow(us_model), nrow(canada_common))
)

cat("\n=== Data Summary ===\n")
print(window_summary)

cat("\n=== Canada Data (first 6 rows) ===\n")
print(head(canada_common))

cat("\n=== US Data (first 6 rows) ===\n")
print(head(us_common))

# ============================================================================
# 9. Save metadata
# ============================================================================
save(
  canada_common,
  us_common,
  file = file.path(data_processed_dir, "wrangled_data.RData")
)

cat("\n Data wrangling complete!")
cat("\nKey files saved:")
cat("\n  - ", canada_common_path)
cat("\n  - ", us_common_path)
cat("\n  - ", file.path(data_processed_dir, "wrangled_data.RData"), "\n")
