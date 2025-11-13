# --- LIBRARIES ---
library(tidymodels)
library(tidyverse)
library(vroom)

# --- LOAD DATA ---
trainData <- vroom("GGG_Comp/train.csv", show_col_types = FALSE)|>
  mutate(type = as.factor(type))
testData <- vroom("GGG_Comp/test.csv", show_col_types = FALSE)

# --- RECIPE ---
my_recipe <- recipe(type ~ ., data = trainData) %>%
  step_normalize(all_numeric_predictors())


# --- MODEL ---
svmRad <- svm_rbf(
  rbf_sigma = tune(),
  cost = tune()
) |>
  set_mode("classification") |>
  set_engine("kernlab")

# ---- WORKFLOW ---
my_workflow <- workflow() |>
  add_model(svmRad) |>
  add_recipe(my_recipe)

# --- TUNIG GRID ---
tuning_grid <- grid_regular(
  cost(range = c(-10, 10)),         # log2 scale
  rbf_sigma(range = c(-5, 5)),    # you forgot to grid over sigma too!
  levels = 7
)

# --- CROSS-VALIDATION ---
folds <- vfold_cv(trainData, v = 10, repeats = 2)

# --- TUNE MODEL ---
CV_results <- my_workflow |>
  tune_grid(
    resamples = folds,
    grid = tuning_grid,
    metrics = metric_set(roc_auc, accuracy, sensitivity, specificity)
  )

# --- SELECT BEST ---
best_tune <- CV_results |>
  select_best(metric = "accuracy")

# --- FINAL WORKFLOW  ---
final_wf <- finalize_workflow(
  my_workflow,
  best_tune
)

# --- FIT ---
final_fit <- final_wf |>
  fit(data = trainData)

# --- PREDICT ---
preds <- predict(final_fit, new_data = testData, type = "class") %>%
  rename(type = .pred_class)

# --- BIND WITH ID ---
preds <- bind_cols(
  id = testData$id,
  preds
)

# --- WRITE SUBMISSION FILE ---
vroom_write(
  x = preds,
  file = "/Users/isaacrands/Documents/Stats/Stat_348/GGG_Comp/Submissions/SVM_GGG_submission.csv",
  delim = ","
)
