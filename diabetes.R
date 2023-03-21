setwd("Diabetes Analysis/")

# install.packages("tidymodels")
# install.packages("glmnet")
# install.packages("ranger")

library(glmnet)
library(ggcorrplot)
library(tidyverse)
library(readxl)
library(stringr)
library(stringi)
library(tidymodels)

diab <- read_csv("diabetes.csv", na = c(" ", "NA", "", NA)) |> select(-id) |> janitor::clean_names()
glimpse(diab)

## Check for missing data and values in our data
diab_na <- is.na(diab)

sum(diab_na)

## Shows that there are no missing values in the data as a whole
## Check and remove outliers
## Glucose, BMI, Insulin, Blood pressure and Skin thickness cannot have value of 0
diab <- diab |> 
  mutate(insulin = ifelse(insulin == 0, NaN, insulin)) |> 
  mutate(bmi = ifelse(bmi == 0, NaN, bmi)) |> 
  mutate(blood_pressure = ifelse(blood_pressure == 0, NaN, blood_pressure)) |> 
  mutate(skin_thickness = ifelse(skin_thickness == 0, NaN, skin_thickness)) |> 
  mutate(glucose = ifelse(glucose == 0, NaN, glucose))


diab_na <- is.na(diab)

sum(diab_na)

## Check for missing values in each column
sapply(diab, function(x) sum(is.na(x)))

## Check for outlier using boxplots and histograms
diab_longer <- diab |> select(-outcome) |> pivot_longer(everything(), names_to = "names", values_to = "values")

diab_longer |> ggplot(mapping = aes(values)) +
  geom_boxplot() +
  facet_wrap(~names, scales = "free", nrow = 3, ncol = 3) +
  coord_flip()

diab_longer |> ggplot(mapping = aes(values)) +
  geom_histogram(color = "red", fill = "white") +
  facet_wrap(~names, scales = "free", nrow = 3, ncol= 3)

## We check how different variables relate to each other and to the presence of diabetes using correlation
diab_corr <- round(cor(na.omit(x_diab_outlier)),3)
diab_pmat <- round(cor_pmat(na.omit(x_diab_outlier)),3)

ggcorrplot(
  diab_corr,
  method = "square",
  show.diag = TRUE,
  # p.mat = diab_pmat,
  hc.order = TRUE,
  type = "full",
  insig = "blank",
  lab = TRUE
)


## Calculating mahalanobis distance to remove outliers
diab[,c(1,3,4,5,6,7,8)]

diab <- na.omit(diab)
mah_dist <- mahalanobis(na.omit(diab)[,1:8], colMeans(na.omit(diab)[,1:8]), cov(na.omit(diab)[,1:8]))
chi_cutoff <- qchisq(0.95, df=ncol(na.omit(diab)[,1:8]))

x_diab_outlier <- diab |> mutate(mah_dist = round(mah_dist, 2)) |> filter(mah_dist < chi_cutoff) |> select(-mah_dist)

x_diab_longer <- x_diab_outlier |> pivot_longer(cols = -outcome, names_to = "names", values_to = "values")

diabetes <- as.factor(x_diab_outlier$outcome)

x_diab_longer |> ggplot(mapping = aes(values, fill = as.factor(outcome))) +
  geom_histogram(color = "red") +
  facet_wrap(~names, scales = "free", nrow = 2, ncol= 4)


x_diab_outlier$outcome<- as.factor(x_diab_outlier$outcome)
levels(x_diab_outlier$outcome)

x_diab_outlier$outcome <- relevel(x_diab_outlier$outcome, ref = "1")
levels(x_diab_outlier$outcome)


#### Preparing our dataset for modelling #####
## Splitting our dataset into train and split using tidymodels initial_split function

diab_split <- x_diab_outlier |> initial_split(prop = 3/4, strata = outcome)
diab_test <- testing(diab_split)
diab_train <- training(diab_split)

nrow(diab_test)
nrow(diab_train)

## Now fitting our logistics regression function using the model function logistic_reg() 
diab_logistic_reg <- logistic_reg() |> 
  set_engine("glm") |> 
  set_mode("classification")

diab_logistic_fit <- diab_logistic_reg |> 
  fit(outcome ~ ., data = diab_train)

## tidy your data
broom::tidy(diab_logistic_fit) |> filter(p.value < 0.05)

## Here we can see that glucose, bmi and dpf have a huge influence of a persons chance of having diabetes

#### Testing our prediction #####
pred_class <- predict(diab_logistic_fit, new_data = diab_test, type = "class")
pred_class[1:5,]

pred_prob <- predict(diab_logistic_fit, new_data = diab_test, type = "prob")
pred_prob[1:5,]

#### Evaluate the model ####
diab_res <- diab_test  |> select(outcome) |> bind_cols(pred_class, pred_prob)

## using confusion matrix
confusion_matrix <- conf_mat(diab_res, truth = outcome, estimate = .pred_class)
confusion_matrix |> autoplot("heatmap")
confusion_matrix |> summary()

## using accuracy
accuracy(data = diab_res, truth = outcome, estimate = .pred_class)

## using sensitivity
sens(data = diab_res, truth = outcome, estimate = .pred_class)

## using specificity
spec(data = diab_res, truth = outcome, estimate = .pred_class)

## Our model is pretty good at predicting those without diabetes

## using precision
precision(data = diab_res, truth = outcome, estimate = .pred_class)

## using recall
recall(data = diab_res, truth = outcome, estimate = .pred_class)

## using f-meassure
f_meas(data = diab_res, truth = outcome, estimate = .pred_class)

## using ROC-AUC 
diab_res
roc_auc(data = diab_res, truth = outcome, estimate = .pred_1)

## Plotting ROC-AUC curve
diab_res_auc <- diab_res |> roc_curve(truth = outcome, estimate = .pred_1) |> mutate(model_name = "BLR")

diab_res_auc |> autoplot()

#### Using Penalized Logistic Regression Models ####
## Partition your training data into training and validation data using the validation_split() function

set.seed(234)
val_set <- validation_split(diab_train, prop = 0.8, strata = outcome)
val_set

## building the model
diab_lr_mod <- logistic_reg(penalty = tune(), mixture = 1) |> 
  set_engine("glmnet")

## the tune() function denotes that we will eventually tune our model hyperparameter to find the the best penalty for our feature selection
## setting mixture equals to 1 will potentially remove any noisy independent variables and produce a better model performance

## creating a recipe for data preprocessing
diab_lr_recip <- recipe(outcome ~ ., data = diab_train) |> 
  step_dummy(all_nominal_predictors()) |> 
  step_zv(all_predictors()) |>## removes variables that cannot be centered and scaled i.e variables with only one value(all zeros or all ones or anything)
  step_normalize(all_predictors()) ## centers and scales all numeric predictors

## creating a workflow
diab_lr_workflow <- workflow() |> 
  add_model(diab_lr_mod) |> 
  add_recipe(diab_lr_recip)
  

## grid for tuning our model hyper parameter(penalty)
diab_lr_grid <- tibble(penalty = 10^seq(-4, -1, length.out = 30))

## training and tuning our model
diab_lr_res <- diab_lr_workflow |> 
  tune_grid(val_set, grid = diab_lr_grid, control = control_grid(save_pred = TRUE), metrics = metric_set(roc_auc))

## plotting our area under the curve graph
diab_lr_plot <- diab_lr_res |> 
  collect_metrics() |> 
  ggplot(aes(penalty, mean)) +
  geom_point() +
  geom_line() +
  scale_x_log10(labels = scales::label_number())

diab_lr_plot

diab_top_models <- diab_lr_res |> 
  show_best("roc_auc") |> 
  arrange(penalty) |> 
  slice(4)

diab_top_models

diab_lr_auc <- diab_lr_res |> 
  collect_predictions(parameters = diab_top_models) |> 
  roc_curve(outcome, .pred_1) |> 
  mutate(model_name = "Penalized Logistics Regression") 

diab_lr_auc |> autoplot()

models_eva <- roc_auc(data = diab_lr_res |> collect_predictions(parameters = diab_top_models), truth = outcome, estimate = .pred_1) |> 
  mutate(model_name = "PLR") |> 
  bind_rows(roc_auc(data = diab_res, truth = outcome, estimate = .pred_1) |> mutate(model_name = "BLR"))

## Using Random Forest Model to Predict our dataset
cores <- parallel::detectCores()
cores

diab_rf_mod <- rand_forest(mtry = tune(), min_n = tune(), trees = 1000) |> 
  set_engine("ranger", num.threads = cores) |> 
  set_mode("classification")

diab_rf_recip <- recipe(outcome ~ ., data = diab_train) |> 
  step_normalize(all_predictors())

diab_rf_workflow <- workflow() |> 
  add_model(diab_rf_mod) |> 
  add_recipe(diab_rf_recip)
diab_rf_workflow

## what to tune
extract_parameter_set_dials(rf_mod)


## Tuning our model hyperparameters
set.seed(234)
diab_rf_res <- diab_rf_workflow |> 
  tune_grid(val_set, grid = 20, control = control_grid(save_pred = TRUE), metrics = metric_set(roc_auc))

diab_rf_res |> show_best(metric = "roc_auc")

diab_rf_res |> collect_metrics()
autoplot(diab_rf_res)

diab_rf_res |> select_best()

diab_rf_res |> collect_predictions()

diab_rf_auc <- diab_rf_res |> collect_predictions(parameters = diab_rf_res |> select_best()) |> roc_curve(outcome, .pred_1) |> 
  mutate(model_name = "random_forest")


full_model <- bind_rows(diab_lr_auc, diab_rf_auc, diab_res_auc)

full_model |> ggplot(aes(x = 1 - specificity, y = sensitivity, col = model_name)) +
  geom_path(linewidth = 1.5, alpha = 0.8) +
  geom_abline(lty = 3) +
  coord_equal() +
  scale_color_viridis_d(option = "plasma", end = .6)



#### Plotting a last fit regression model on our logistics regression model ####
diab_last_fit <- diab_logistic_reg |> 
  last_fit(outcome ~ ., split = diab_split)

diab_last_fit |> collect_metrics()

diab_last_fit_result <- diab_last_fit |> collect_predictions()

conf_mat(data = diab_last_fit_result, truth = outcome, estimate = .pred_class) |> summary()

## creating custom metrics function
last_fit_metrics <- metric_set(accuracy, roc_auc, sens, spec)
last_fit_metrics(data = diab_last_fit_result, truth = outcome, estimate = .pred_class, .pred_1)

diab_last_fit |> collect_predictions() |> roc_curve(truth = outcome, .pred_1) |> autoplot()
