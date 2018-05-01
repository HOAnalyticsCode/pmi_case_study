# ENVIRONMENT #
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
import json
from sklearn.model_selection import train_test_split
from sklearn.model_selection import learning_curve
from sklearn.model_selection import validation_curve
import matplotlib.pyplot as plt
import project_functions as pf

# DATA and REFORMATTING #
sales_df = pd.read_csv("sales_granular.csv")
surroundings_json = json.load(open("Surroundings.json"))

sales_df = pf.sales_df_reformatting(sales_df)

# List of amenities from json file
amenities_ls = list(surroundings_json[1]['surroundings'].keys())

# Store codes in the json file
store_codes_json = [surroundings_json[n]['store_code'] for n in range(len(surroundings_json))]

# TARGET, FEATURES and MODELLING datasets
target_variables_df = pf.create_target_variables_df(sales_df)

explanatory_variables_df = pf.create_amenities_count_df(store_codes_json,
                                                        surroundings_json,
                                                        amenities_ls)

# Target and explanatory merge in order to use R formula syntax
modelling_df = pd.merge(target_variables_df,
                        explanatory_variables_df,
                        left_index=True,
                        right_index=True)

# MODELLING #

# Due to high dimensionality will try the following methods assuming underlying poisson family:
# GB
# Random Forest
# PCR (Principal Component Regression) - separate treatment
# LASSO - separate treatment

# NOTE: Unfortunately comparative analysis is only in R
# Below GB model fit. Note sklearn implementation doesn't seem to allow 
# for a choice of family of distributions
X_train, X_test, y_train, y_test = train_test_split(modelling_df[amenities_ls], modelling_df['AVG_SALES'])
gbrt = GradientBoostingRegressor(n_estimators=100)
gbrt.fit(X_train, y_train)
y_pred = gbrt.predict(X_test)
gbrt_MSSR = sum((y_pred - y_test) ** 2)

no_model_MSSR = sum((np.mean(y_train) - y_test) ** 2)
# No real predictive power
gbrt_MSSR / no_model_MSSR

feature_importances = dict(zip(amenities_ls, gbrt.feature_importances_))
sorted(feature_importances, key=feature_importances.get)

# The score suggest overfitting with sklearn base parameters
# Perhaps grid search could help
gbrt.score(X_train, y_train)
gbrt.score(X_test, y_test)

# Learning curve
train_sizes, train_scores, test_scores = learning_curve(gbrt, X_train, y_train, cv=10)

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

# Definitely a model with room for improvement
plt.plot(train_sizes, train_mean,
         color='blue', marker='o',
         markersize=5, label='training accuracy')

plt.fill_between(train_sizes,
                 train_mean + train_std,
                 train_mean - train_std,
                 alpha=0.15, color='blue')

plt.plot(train_sizes, test_mean,
         color='green', linestyle='--',
         marker='s', markersize=5,
         label='validation accuracy')

plt.fill_between(train_sizes,
                 test_mean + test_std,
                 test_mean - test_std,
                 alpha=0.15, color='green')

plt.grid()
plt.xlabel('Number of training samples')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.ylim([-1, 2])
plt.tight_layout()
plt.show()

# Validation curve over shrinkage
param_range = [0.001, 0.01, 0.1, 0.3]
train_scores, test_scores = validation_curve(
                estimator=gbrt,
                X=X_train,
                y=y_train,
                param_name='learning_rate',
                param_range=param_range,
                cv=10)

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

plt.plot(param_range, train_mean,
         color='blue', marker='o',
         markersize=5, label='training accuracy')

plt.fill_between(param_range, train_mean + train_std,
                 train_mean - train_std, alpha=0.15,
                 color='blue')

plt.plot(param_range, test_mean,
         color='green', linestyle='--',
         marker='s', markersize=5,
         label='validation accuracy')

plt.fill_between(param_range,
                 test_mean + test_std,
                 test_mean - test_std,
                 alpha=0.15, color='green')

plt.grid()
plt.xscale('log')
plt.legend(loc='lower right')
plt.xlabel('Parameter learning_rate')
plt.ylabel('Accuracy')
plt.ylim([-2, 1.5])
plt.tight_layout()
plt.show()

# The model is as bad as it gets from a predictive perspective.
# Back to bias, variance, data features and deeper data analysis.
