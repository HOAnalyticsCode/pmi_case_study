# ENVIRONMENT #
library(glmnet)
library(gbm)
library(randomForest)
library(boot)
library(jsonlite)
library(lubridate)
source("project_functions.R")

# DATA and REFORMATTING #
sales_df <- read.csv("sales_granular.csv")
surroundings_json <- jsonlite::read_json("Surroundings.json")

sales_df <- sales_df_reformatting(sales_df)

# List of amenities from json file
amenities_ls <- names(surroundings_json[[1]]$surroundings)
# Store codes in the json file
store_codes_json <-
  sapply(1:length(surroundings_json), function(x)
    surroundings_json[[x]]$store_code)

# TARGET, FEATURES and MODELLING datasets
target_variables_df <- create_target_variables_df(sales_df)

explanatory_variables_df <-
  create_amenities_count_df(
    store_codes = target_variables_df$STORE_CODE,
    surroundings_json,
    amenities_ls,
    store_codes_json,
    reviews_weighted = F
  )


# Target and explanatory merge in order to use R formula syntax
modelling_df <- merge(
  target_variables_df,
  explanatory_variables_df,
  by.x = "STORE_CODE",
  by.y = "STORE_CODE",
  all.y = T
)

modelling_df <- modelling_df_preprocessing(modelling_df,
                                           amenities_ls,
                                           cor_filter = F)

# MODELLING #

# Due to high dimensionality will try the following methods assuming underlying poisson family:
# GB
# Random Forest
# PCR (Principal Component Regression) - separate treatment
# LASSO - separate treatment

create_mssr_comparisons_plot(
  formula = AVG_SALES ~ .,
  no_modelling_vars = c(
    "STORE_CODE",
    "SALES_VOLATILITY",
    "WEEKS_ONSALE",
    "NOISY_POS",
    "n_amenities"
  ),
  family = "poisson",
  data = modelling_df,
  folds = 50,
  ratio = 0.8
)

# GBM FOCUS
gbm_model <-
  gbm::gbm(
    AVG_SALES ~ .,
    data = modelling_df[, setdiff(
      colnames(modelling_df),
      c(
        "STORE_CODE",
        "SALES_VOLATILITY",
        "WEEKS_ONSALE",
        "NOISY_POS",
        "n_amenities"
      )
    )],
    distribution = "poisson",
    shrinkage = 0.05
  )

gbm_summary <- summary(gbm_model)
new_formula <-
  as.formula(paste("AVG_SALES ~ +", paste(gbm_summary$var[gbm_summary$rel.inf > 1],
                                          collapse = " + ")))

plot(modelling_df$AVG_SALES, type = "l")
lines(
  predict(gbm_model, modelling_df, type = "response",
          n.trees = 100),
  col = "green",
  lty = 2
)

# GLM model with varibles identified by gbm
glm_model <- glm(new_formula, data = modelling_df,
                 family = "poisson")

summary(glm_model)
hist(residuals(glm_model))

plot(modelling_df$AVG_SALES, type = "l")
lines(glm_model$fitted.values, col = "green", lty = 2)

cv_glm <- cv.glm(modelling_df, glm_model)
cv_glm$delta

# Only avg performance
no_model <- glm(AVG_SALES ~ 1,
                data = modelling_df,
                family = "poisson")

cv_no_model <- cv.glm(modelling_df, no_model)
cv_no_model$delta

# All variables model
all_model <- glm(AVG_SALES ~ .,
                 data = modelling_df[, c("AVG_SALES", amenities_ls)],
                 family = "poisson")

summary(all_model)

cv_all_model <-
  cv.glm(modelling_df[, c("AVG_SALES", amenities_ls)], all_model)
cv_all_model$delta

