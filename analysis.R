library(forecast)
library(readr)

# -------------------------
# 1. Load data
# -------------------------
can <- read_csv("data_processed/canada_model_common_window.csv")
us  <- read_csv("data_processed/us_model_common_window.csv")

# Keep only complete rows
can <- can[complete.cases(can), ]
us  <- us[complete.cases(us), ]
can <- head(can, -4)
us <- head(us, -1)


# -------------------------
# 2. Train / holdout split
# -------------------------
h <- 24

n_can <- nrow(can)
train_can <- can[1:(n_can - h), ]
test_can  <- can[(n_can - h + 1):n_can, ]

n_us <- nrow(us)
train_us <- us[1:(n_us - h), ]
test_us  <- us[(n_us - h + 1):n_us, ]

# -------------------------
# 3. Diagnostics
# -------------------------
# ACF/PACF for inflation
acf(train_can$infl, main = "Canada Inflation ACF")
pacf(train_can$infl, main = "Canada Inflation PACF")

acf(train_us$infl, main = "US Inflation ACF")
pacf(train_us$infl, main = "US Inflation PACF")

# CCFs for Canada
ccf(train_can$infl, train_can$oil_wti_l1,   main = "Canada: infl vs oil")
ccf(train_can$infl, train_can$ippi_can_l1,  main = "Canada: infl vs IPPI")
ccf(train_can$infl, train_can$gdp_can_l1,   main = "Canada: infl vs GDP")
ccf(train_can$infl, train_can$unemp_can_l1, main = "Canada: infl vs unemployment")
ccf(train_can$infl, train_can$exrate_can_l1, main = "Canada: infl vs exchange rate")

# CCFs for US
ccf(train_us$infl, train_us$oil_wti_l1, main = "US: infl vs oil")
ccf(train_us$infl, train_us$ppi_us_l1,  main = "US: infl vs PPI")
ccf(train_us$infl, train_us$gdp_us_l1,  main = "US: infl vs GDP")
ccf(train_us$infl, train_us$u6_us_l1,   main = "US: infl vs unemployment")
ccf(train_us$infl, train_us$mxp_us_l1,  main = "US: infl vs MXP")

# -------------------------
# 4. Helper functions
# -------------------------
rmse <- function(actual, pred) {
  sqrt(mean((actual - pred)^2, na.rm = TRUE))
}

mean_fc <- function(train, test) {
  rep(mean(train, na.rm = TRUE), length(test))
}

mean_insample <- function(train) {
  rep(mean(train, na.rm = TRUE), length(train))
}

persistence_fc <- function(train, test) {
  preds <- numeric(length(test))
  last <- tail(train, 1)
  for (i in seq_along(test)) {
    preds[i] <- last
    last <- test[i]
  }
  preds
}

persistence_insample <- function(train) {
  c(NA, train[-length(train)])
}

ets_fc <- function(train, test) {
  preds <- numeric(length(test))
  y <- train
  for (i in seq_along(test)) {
    fit <- ets(y)
    preds[i] <- as.numeric(forecast(fit, h = 1)$mean[1])
    y <- c(y, test[i])
  }
  preds
}

ets_insample <- function(train) {
  fit <- ets(train)
  as.numeric(fitted(fit))
}

arima_fc <- function(train, test, order) {
  preds <- numeric(length(test))
  y <- train
  for (i in seq_along(test)) {
    fit <- Arima(y, order = order)
    preds[i] <- as.numeric(forecast(fit, h = 1)$mean[1])
    y <- c(y, test[i])
  }
  preds
}

arima_insample <- function(train, order) {
  fit <- Arima(train, order = order)
  as.numeric(fitted(fit))
}

# ARIMAX 1-step forecasts to avoid interval issues
arimax_fc <- function(train_y, test_y, train_x, test_x, order = NULL) {
  preds <- numeric(length(test_y))
  y <- train_y
  x <- train_x
  
  for (i in seq_along(test_y)) {
    fit <- if (is.null(order)) {
      auto.arima(y, xreg = x, seasonal = FALSE, stepwise = TRUE, approximation = FALSE)
    } else {
      Arima(y, order = order, xreg = x)
    }
    
    pr <- predict(fit, n.ahead = 1, newxreg = test_x[i, , drop = FALSE])
    preds[i] <- as.numeric(pr$pred[1])
    
    y <- c(y, test_y[i])
    x <- rbind(x, test_x[i, , drop = FALSE])
  }
  
  preds
}

arimax_insample <- function(train_y, train_x, order = NULL) {
  fit <- if (is.null(order)) {
    auto.arima(train_y, xreg = train_x, seasonal = FALSE, stepwise = TRUE, approximation = FALSE)
  } else {
    Arima(train_y, order = order, xreg = train_x)
  }
  as.numeric(fitted(fit))
}

# -------------------------
# 5. ARIMA candidates
# -------------------------
arima_candidates <- list(
  c(1,0,1),
  c(2,0,1),
  c(1,0,2)
)

# Canada ARIMA candidates
can_arima_rmse <- sapply(arima_candidates, function(ord) {
  preds <- arima_fc(train_can$infl, test_can$infl, ord)
  rmse(test_can$infl, preds)
})

best_can_arima_idx <- which.min(can_arima_rmse)
best_can_arima_order <- arima_candidates[[best_can_arima_idx]]
best_can_arima_pred <- arima_fc(train_can$infl, test_can$infl, best_can_arima_order)

# US ARIMA candidates
us_arima_rmse <- sapply(arima_candidates, function(ord) {
  preds <- arima_fc(train_us$infl, test_us$infl, ord)
  rmse(test_us$infl, preds)
})

best_us_arima_idx <- which.min(us_arima_rmse)
best_us_arima_order <- arima_candidates[[best_us_arima_idx]]
best_us_arima_pred <- arima_fc(train_us$infl, test_us$infl, best_us_arima_order)

# -------------------------
# 6. ARIMAX candidates
# -------------------------
# Canada: based on CCFs and results
x_train_can_oil <- cbind(train_can$oil_wti_l1)
x_test_can_oil  <- cbind(test_can$oil_wti_l1)

x_train_can_oil_ippi <- cbind(
  train_can$oil_wti_l1,
  train_can$ippi_can_l1
)
x_test_can_oil_ippi <- cbind(
  test_can$oil_wti_l1,
  test_can$ippi_can_l1
)

can_arimax_specs <- list(
  oil = list(train = x_train_can_oil, test = x_test_can_oil),
  oil_ippi = list(train = x_train_can_oil_ippi, test = x_test_can_oil_ippi)
)

can_arimax_rmse <- sapply(can_arimax_specs, function(spec) {
  preds <- arimax_fc(train_can$infl, test_can$infl, spec$train, spec$test)
  rmse(test_can$infl, preds)
})

best_can_arimax_name <- names(which.min(can_arimax_rmse))
best_can_arimax_train <- can_arimax_specs[[best_can_arimax_name]]$train
best_can_arimax_test  <- can_arimax_specs[[best_can_arimax_name]]$test
best_can_arimax_pred  <- arimax_fc(train_can$infl, test_can$infl, best_can_arimax_train, best_can_arimax_test)

# US
x_train_us_oil_ppi <- cbind(
  train_us$oil_wti_l1,
  train_us$ppi_us_l1
)
x_test_us_oil_ppi <- cbind(
  test_us$oil_wti_l1,
  test_us$ppi_us_l1
)

x_train_us_oil_ppi_mxp <- cbind(
  train_us$oil_wti_l1,
  train_us$ppi_us_l1,
  train_us$mxp_us_l1
)
x_test_us_oil_ppi_mxp <- cbind(
  test_us$oil_wti_l1,
  test_us$ppi_us_l1,
  test_us$mxp_us_l1
)

x_train_us_oil_ppi_mxp_u6 <- cbind(
  train_us$oil_wti_l1,
  train_us$ppi_us_l1,
  train_us$mxp_us_l1,
  train_us$u6_us_l1
)
x_test_us_oil_ppi_mxp_u6 <- cbind(
  test_us$oil_wti_l1,
  test_us$ppi_us_l1,
  test_us$mxp_us_l1,
  test_us$u6_us_l1
)

us_arimax_specs <- list(
  oil_ppi = list(train = x_train_us_oil_ppi, test = x_test_us_oil_ppi),
  oil_ppi_mxp = list(train = x_train_us_oil_ppi_mxp, test = x_test_us_oil_ppi_mxp),
  oil_ppi_mxp_u6 = list(train = x_train_us_oil_ppi_mxp_u6, test = x_test_us_oil_ppi_mxp_u6)
)

us_arimax_rmse <- sapply(us_arimax_specs, function(spec) {
  preds <- arimax_fc(train_us$infl, test_us$infl, spec$train, spec$test)
  rmse(test_us$infl, preds)
})

best_us_arimax_name <- names(which.min(us_arimax_rmse))
best_us_arimax_train <- us_arimax_specs[[best_us_arimax_name]]$train
best_us_arimax_test  <- us_arimax_specs[[best_us_arimax_name]]$test
best_us_arimax_pred  <- arimax_fc(train_us$infl, test_us$infl, best_us_arimax_train, best_us_arimax_test)

# -------------------------
# 7. Out-of-sample RMSE
# -------------------------
can_mean_pred <- mean_fc(train_can$infl, test_can$infl)
can_persist_pred <- persistence_fc(train_can$infl, test_can$infl)
can_ets_pred <- ets_fc(train_can$infl, test_can$infl)

results_can_out <- data.frame(
  Model = c("Mean", "Persistence", "ETS", "ARIMA", "ARIMAX"),
  Out_of_sample_RMSE = c(
    rmse(test_can$infl, can_mean_pred),
    rmse(test_can$infl, can_persist_pred),
    rmse(test_can$infl, can_ets_pred),
    rmse(test_can$infl, best_can_arima_pred),
    rmse(test_can$infl, best_can_arimax_pred)
  )
)

us_mean_pred <- mean_fc(train_us$infl, test_us$infl)
us_persist_pred <- persistence_fc(train_us$infl, test_us$infl)
us_ets_pred <- ets_fc(train_us$infl, test_us$infl)

results_us_out <- data.frame(
  Model = c("Mean", "Persistence", "ETS", "ARIMA", "ARIMAX"),
  Out_of_sample_RMSE = c(
    rmse(test_us$infl, us_mean_pred),
    rmse(test_us$infl, us_persist_pred),
    rmse(test_us$infl, us_ets_pred),
    rmse(test_us$infl, best_us_arima_pred),
    rmse(test_us$infl, best_us_arimax_pred)
  )
)

# -------------------------
# 8. In-sample RMSE
# -------------------------
can_mean_fit <- mean_insample(train_can$infl)
can_persist_fit <- persistence_insample(train_can$infl)
can_ets_fit <- ets_insample(train_can$infl)
can_arima_fit <- arima_insample(train_can$infl, best_can_arima_order)
can_arimax_fit <- arimax_insample(train_can$infl, best_can_arimax_train)

results_can_in <- data.frame(
  Model = c("Mean", "Persistence", "ETS", "ARIMA", "ARIMAX"),
  In_sample_RMSE = c(
    rmse(train_can$infl, can_mean_fit),
    rmse(train_can$infl, can_persist_fit),
    rmse(train_can$infl, can_ets_fit),
    rmse(train_can$infl, can_arima_fit),
    rmse(train_can$infl, can_arimax_fit)
  )
)

us_mean_fit <- mean_insample(train_us$infl)
us_persist_fit <- persistence_insample(train_us$infl)
us_ets_fit <- ets_insample(train_us$infl)
us_arima_fit <- arima_insample(train_us$infl, best_us_arima_order)
us_arimax_fit <- arimax_insample(train_us$infl, best_us_arimax_train)

results_us_in <- data.frame(
  Model = c("Mean", "Persistence", "ETS", "ARIMA", "ARIMAX"),
  In_sample_RMSE = c(
    rmse(train_us$infl, us_mean_fit),
    rmse(train_us$infl, us_persist_fit),
    rmse(train_us$infl, us_ets_fit),
    rmse(train_us$infl, us_arima_fit),
    rmse(train_us$infl, us_arimax_fit)
  )
)

# -------------------------
# 9. Final comparison tables
# -------------------------
results_can <- merge(results_can_in, results_can_out, by = "Model")
results_us  <- merge(results_us_in, results_us_out, by = "Model")

results_can <- results_can[order(results_can$Out_of_sample_RMSE), ]
results_us  <- results_us[order(results_us$Out_of_sample_RMSE), ]

print("Best Canada ARIMA order:")
print(best_can_arima_order)

print("Best Canada ARIMAX spec:")
print(best_can_arimax_name)

print("Best US ARIMA order:")
print(best_us_arima_order)

print("Best US ARIMAX spec:")
print(best_us_arimax_name)

print("Canada results:")
print(results_can)

print("US results:")
print(results_us)

# -------------------------
# 10. Save outputs
# -------------------------
write.csv(results_can, "results_can_final.csv", row.names = FALSE)
write.csv(results_us, "results_us_final.csv", row.names = FALSE)