library(tidyverse)
library(tidymodels)
library(embed)
library(vroom)
library(workflows)
library(ranger)
library(themis)

# --- READ DATA ---
trainData <- vroom("GGG_Comp/train.csv", show_col_types = FALSE)
testData <- vroom("GGG_Comp/test.csv", show_col_types = FALSE)

# --- RECIPE ---
my_recipe <- recipe(type ~ ., data = trainData) %>%
  update_role(id, new_role = "ID") %>%  # exclude ID from predictors
  step_dummy(all_nominal_predictors()) %>%
  step_normalize(all_numeric_predictors())

# --- MODEL SPEC ---
rf_model <- rand_forest(
  mtry = tune(),
  min_n = tune(),
  trees = 1000
) %>%
  set_mode("classification") %>%
  set_engine("ranger")

# --- WORKFLOW ---
rf_wf <- workflow() %>%
  add_model(rf_model) %>%
  add_recipe(my_recipe)

# --- GRID ---
tuning_grid <- grid_regular(
  mtry(range = c(1, 5)),  # number of predictors randomly sampled at each split
  min_n(), 
  levels = 3
)

# --- FOLDS ---
folds <- vfold_cv(trainData, v = 10, repeats = 1, strata = type)

# --- TUNE ---
cv_results <- tune_grid(
  rf_wf,
  resamples = folds,
  grid = tuning_grid,
  metrics = metric_set(roc_auc)
)

best_tune <- cv_results %>% select_best(metric = "roc_auc")

# --- FINAL FIT ---
rf_fit <- rf_wf %>%
  finalize_workflow(best_tune) %>%
  fit(data = trainData)

# --- PREDICT ON TEST DATA ---
preds <- predict(rf_fit, new_data = testData, type = "class") %>%
  rename(type = .pred_class)

# --- BIND WITH ID ---
preds <- bind_cols(
  id = testData$id,
  preds
)

# --- WRITE SUBMISSION FILE ---
vroom_write(
  x = preds,
  file = "/Users/isaacrands/Documents/Stats/Stat_348/GGG_Comp/Submissions/RF_GGG_submission.csv",
  delim = ","
)
