#https://business-science.github.io/modeltime.gluonts/reference/
#https://business-science.github.io/modeltime.gluonts/reference/install_gluonts.html
#https://business-science.github.io/modeltime/reference/
#https://business-science.github.io/modeltime/reference/m750.html
#https://business-science.github.io/modeltime.gluonts/articles/managing-envs.html
#https://github.com/business-science/modeltime/issues/5


# Setup ----
library(tidymodels)
library(workflowsets)
library(modeltime)
library(modeltime.gluonts)
library(modeltime.ensemble)
library(timetk)
library(parallel)
library(stringr)
library(stacks)

# Split Data ----
a <- m750 %>%
  mutate(id = str_trim(id)) %>%
  filter(id == "M750")
a$id <- as.factor(a$id)
a <- a %>%
  mutate(temp = rnorm(nrow(a), mean = 20, sd = 5))
splits <- time_series_split(
  data = a,
  assess = 30,
  cumulative = TRUE
)

# Define the recipe ----
recipe_spec <- recipe(value ~ date + temp, data = a)

# GluonTS Models (No tuning required) ----
# DeepAR ----
fit_deepar_gluonts <- deep_ar(
  id = "id",
  freq = "M",
  prediction_length = 30,
  lookback_length = 30*3,
  epochs = 3
) %>%
  set_engine("gluonts_deepar") %>%
  fit(value ~ date + id + temp, data = training(splits))

# GP Forecaster ----
fit_gp_forecaster <- gp_forecaster(
  id = "id",
  freq = "M",
  prediction_length = 30,
  lookback_length = 30*3,
  epochs = 3*2
) %>%
  set_engine("gluonts_gp_forecaster")%>%
  fit(value ~ date + id + temp, data = training(splits))

# Deep State ----
fit_deep_state <- deep_state(
  id = "id",
  freq = "M",
  prediction_length = 30,
  lookback_length = 30*3,
  epochs = 3*2
) %>%
  set_engine("gluonts_deepstate")%>%
  fit(value ~ date + id + temp, data = training(splits))

# N-BEATS ----
fit_nbeats <- nbeats(
  id = "id",
  freq = "M",
  prediction_length = 30,
  lookback_length = 30*3,
  epochs = 3*2
) %>%
  set_engine("gluonts_nbeats")%>%
  fit(value ~ date + id + temp, data = training(splits))

# Standard Prophet ----
fit_prophet <- prophet_reg(
  seasonality_yearly = TRUE
) %>%
  set_engine("prophet") %>%
  fit(value ~ date + temp, data = training(splits))

# Standard ARIMA ----
fit_arima <- arima_reg() %>%
  set_engine("auto_arima") %>%
  fit(value ~ date + temp, data = training(splits))

# Prophet Boost Tuning ----
prophet_boost_spec <- prophet_boost(
  seasonality_yearly = TRUE,
  changepoint_num = tune(),
  changepoint_range = tune()
) %>%
  set_engine("prophet_xgboost")

prophet_boost_wf <- workflow() %>%
  add_model(prophet_boost_spec) %>%
  add_recipe(recipe_spec)

prophet_boost_grid <- grid_regular(
  changepoint_num(c(1, 10)),
  changepoint_range(c(0.5, 0.9)),
  levels = 5
)

prophet_boost_tuned <- tune_grid(
  prophet_boost_wf,
  resamples = training(splits),
  grid = prophet_boost_grid,
  control = control_grid(save_pred = TRUE)
)

# ARIMA Boost Tuning ----
arima_boost_spec <- arima_boost(
  non_seasonal_ar = tune(),
  non_seasonal_differences = tune(),
  non_seasonal_ma = tune(),
  seasonal_ar = tune(),
  seasonal_ma = tune(),
  seasonal_differences = tune()
) %>%
  set_engine("auto_arima_xgboost")

arima_boost_wf <- workflow() %>%
  add_model(arima_boost_spec) %>%
  add_recipe(recipe_spec)

arima_boost_grid <- grid_regular(
  non_seasonal_ar(c(0, 5)),
  non_seasonal_ma(c(0, 5)),
  seasonal_ar(c(0, 3)),
  seasonal_ma(c(0, 3)),
  levels = 5
)

arima_boost_tuned <- tune_grid(
  arima_boost_wf,
  resamples = training(splits),
  grid = arima_boost_grid,
  control = control_grid(save_pred = TRUE)
)

# NNETAR Tuning ----
nnetar_spec <- nnetar_reg(
  num_networks = tune()
) %>%
  set_engine("nnetar")

nnetar_wf <- workflow() %>%
  add_model(nnetar_spec) %>%
  add_recipe(recipe_spec)

nnetar_grid <- grid_regular(
  num_networks(c(10, 20, 30)),
  levels = 5
)

nnetar_tuned <- tune_grid(
  nnetar_wf,
  resamples = training(splits),
  grid = nnetar_grid,
  control = control_grid(save_pred = TRUE)
)

# Exponential Smoothing Tuning ----
fit_exp_smooth <- exp_smoothing(
  error = tune(),
  trend = tune(),
  season = tune()
) %>%
  set_engine("ets")

exp_smooth_wf <- workflow() %>%
  add_model(fit_exp_smooth) %>%
  add_recipe(recipe_spec)

exp_smooth_grid <- grid_regular(
  error(c("additive", "multiplicative")),
  trend(c("additive", "multiplicative")),
  season(c("additive", "multiplicative")),
  levels = 5
)

exp_smooth_tuned <- tune_grid(
  exp_smooth_wf,
  resamples = training(splits),
  grid = exp_smooth_grid,
  control = control_grid(save_pred = TRUE)
)

# Seasonal Regression ----
fit_seasonal_reg <- seasonal_reg() %>%
  set_engine("tbats") %>%
  fit(value ~ date + temp, data = training(splits))

# Stacked Model ----
stacked_model <- stacks() %>%
  add_candidates(prophet_boost_tuned) %>%
  add_candidates(arima_boost_tuned) %>%
  add_candidates(nnetar_tuned) %>%
  add_candidates(exp_smooth_tuned) %>%
  add_candidates(fit_seasonal_reg) %>%
  blend_predictions() %>%
  fit_members()

# Calibrate Models ----
calib_gluonts_tbl <- modeltime_table(
  fit_deepar_gluonts,
  fit_gp_forecaster,
  fit_deep_state,
  fit_nbeats,
  fit_prophet,
  fit_arima,
  prophet_boost_tuned %>% finalize_model(),
  arima_boost_tuned %>% finalize_model(),
  nnetar_tuned %>% finalize_model(),
  exp_smooth_tuned %>% finalize_model(),
  fit_seasonal_reg,
  stacked_model  # Include stacked model
) %>%
  modeltime_calibrate(testing(splits), id = "id")

# Global and ID Accuracy ----
calib_gluonts_tbl %>%
  modeltime_accuracy()

accuracy_id_tbl <- calib_gluonts_tbl %>%
  modeltime_accuracy(acc_by_id = TRUE) %>%
  group_by(id) %>%
  slice_min(rmse)

accuracy_id_tbl

# Forecasting ----
calib_gluonts_tbl %>%
  modeltime_forecast(
    new_data = testing(splits),
    actual_data = m750,
    keep_data = TRUE
  ) %>%
  group_by(id) %>%
  plot_modeltime_forecast(.facet_ncol = 1, .plotly_slider = TRUE)

# Residual Analysis ----
residuals_tbl <- calib_gluonts_tbl %>%
  modeltime_residuals()

# Plot residuals
residuals_tbl %>%
  plot_modeltime_residuals(.facet_ncol = 1, .interactive = TRUE)

# Statistical tests for residuals
residual_tests <- residuals_tbl %>%
  modeltime_residuals_test()

# Display residual tests
print(residual_tests)

# Interactive table of accuracy
calib_gluonts_tbl %>%
  table_modeltime_accuracy()

# Final Forecasts & Exploration
calib_gluonts_tbl
