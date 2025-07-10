# type: ignore
# flake8: noqa
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#| echo: true
#| code-fold: true
#| code-summary: "Imports"
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px

# Deeply copying objects
from sklearn.base import clone

# Data source
from sklearn.datasets import fetch_california_housing

# Regression random forest
from sklearn.ensemble import RandomForestRegressor

# For splitting dataset
from sklearn.model_selection import (
  cross_val_score,
  GridSearchCV,
  train_test_split
)

# For composing transformations
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# For preprocessing data
from sklearn.preprocessing import(
  FunctionTransformer, 
  PolynomialFeatures,
  StandardScaler
)

# Linear models 
from sklearn.linear_model import (
  Lasso,
  LinearRegression,
)

# Model evaluation
from sklearn.metrics import (
  root_mean_squared_error,
)

# Regression trees
from sklearn.tree import DecisionTreeRegressor
#
#
#
# Hidden: set data path
from pathlib import Path
data_path = Path() / "slides" / "prediction" / "data"
#
#
#
#
#
#
#
#
#
#
#
#| echo: true
#| code-fold: true
#| code-summary: "Loading data"
# Load data
data = fetch_california_housing(data_home=data_path, as_frame=True)
data_df = data.frame.copy()    
# Split off test test             
train_set, test_set = train_test_split(data_df, test_size = 0.2, random_state= 1)
# Separate the Xs and the labels
X_train = train_set.drop("MedHouseVal", axis=1)
y_train = train_set["MedHouseVal"].copy()
#
#
#
#
#
#
#
#
#
#
#
#
BG_COLOR = "whitesmoke"
FONT_COLOR = "black"
GEO_COLOR = "rgb(201, 201, 201)"
OCEAN_COLOR = "rgb(136, 136, 136)"

fig, ax = plt.subplots(figsize=(14, 6.5))
fig.patch.set_facecolor(BG_COLOR)
fig.patch.set_edgecolor("teal")
fig.patch.set_linewidth(5)
scatter = train_set.plot(
    kind="scatter",
    x="Longitude",
    y="Latitude",
    grid=True,
    s=train_set["Population"] / 100,
    label="Population",
    c="MedHouseVal",
    cmap="BuPu",
    colorbar=False,
    legend=False,
    sharex=False,
    ax=ax
)

# Set the main title
ax.set_title("Geographical distribution of points, scaled by population, color by median house value", loc="left")
# Add colorbar and set its title
cbar = plt.colorbar(scatter.get_children()[0], ax=ax)
cbar.set_label("Median House Value")

plt.show()
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#| echo: true
#| code-fold: true
#| code-summary: "Creating the pipeline"
# Define the ratio transformers
def column_ratio(X):
    return X[:, [0]] / X[:, [1]]
def ratio_name(function_transformer, feature_names_in):
    return ["ratio"]
divider_transformer = FunctionTransformer(
  column_ratio, 
  validate=True, 
  feature_names_out = ratio_name)

# Extracting and dropping features
feat_extr_pipe = ColumnTransformer(
  [
    ('bedroom_ratio', divider_transformer, ['AveBedrms', 'AveRooms']),
    (
      'passthrough', 
      'passthrough', 
      [
        'MedInc', 
        'HouseAge', 
        'AveRooms', 
        'AveBedrms', 
        'Population', 
        'AveOccup',
      ]
    ),
    ('drop', 'drop', ['Longitude', 'Latitude'])
  ]
) 

# Creating polynomials and standardizing
preprocessing = Pipeline(
  [
    ('extraction', feat_extr_pipe),
    ('poly', PolynomialFeatures(include_bias=False)),
    ('scale', StandardScaler()),
  ]
)
#
#
#
#
#
#
#
#
#
#
#
#
preprocessing
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#| echo: true
ols_mod = LinearRegression()          # Create instance
ols_mod.fit(X_train, y_train)         # Fit (run OLS)
ols_mod.predict(X_train.iloc[:5, :])  # Predict for first 5 training obs
```
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#| echo: true
ols_sq_model = Pipeline(
  [
    ("preprocessing", preprocessing),
    ("ols", LinearRegression())
  ]
)
#
#
#
#
#
ols_sq_model
#
#
#
#
#
#
#
#
#
#
#
#| echo: true
ols_sq_model.fit(X_train, y_train);
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#| echo: true
root_mean_squared_error(y_train, ols_sq_model.predict(X_train))
```
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#| echo: true
lasso_model = Pipeline(
  [
    ("preprocessing", clone(preprocessing)),
    ("lasso", Lasso())
  ]
)
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#| echo: true
# Fit model
lasso_model.fit(X_train, y_train)
# Evaluate on training sample
root_mean_squared_error(y_train, lasso_model.predict(X_train))
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#| echo: true
ols_rmse = -cross_val_score(
    ols_sq_model,                           # algorithm
    X_train,
    y_train,
    scoring="neg_root_mean_squared_error",  # which score to compute
    cv=10,                                  # how many folds
)

#
#
#
#
#
#
#
#
#
#| echo: true
pd.Series(ols_rmse).describe().iloc[:3]
#
#
#
#
#
#
ols_rmse
```
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#| echo: true
param_grid_alpha = { 
    "lasso__alpha": np.linspace(0, 1, 11),
}
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#| echo: true
grid_search_alpha = GridSearchCV(
    lasso_model,                           # which predictor/pipeline
    param_grid_alpha,                      # parameter grids
    cv=10,                                 # number of folds
    n_jobs = -1,                           # number of parallel jobs
    scoring="neg_root_mean_squared_error", # which score
)
grid_search_alpha.fit(X_train, y_train);
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#| echo: true
cv_res_alpha = pd.DataFrame(grid_search_alpha.cv_results_)
cv_res_alpha.sort_values(by='rank_test_score')
# Can also flip the score sign to turn into RMSE
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#| echo: true
param_grid = { 
    "preprocessing__poly__degree": [1, 2, 3],
    "lasso__alpha": np.linspace(0, 1, 11),
}
```
#
#
#
#
#| echo: true
#| code-fold: true
#| code-summary: Calling `GridSearchCV`
grid_search = GridSearchCV(
    lasso_model,
    param_grid,
    cv=10,
    n_jobs = -1,
    scoring="neg_root_mean_squared_error",
)

grid_search.fit(X_train, y_train);
#
#
#
#
#
#
#
#
#
#
cv_res = pd.DataFrame(grid_search.cv_results_)
cv_res.sort_values(by='rank_test_score').head(3).loc[:, 
["param_lasso__alpha", "param_preprocessing__poly__degree",
  'mean_test_score', 'std_test_score',
       'rank_test_score']] 
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#| echo: true
#| code-fold: true
#| code-summary: "Recreating preprocessing pipeline"
preprocessing = Pipeline(
  [
    ('extraction', feat_extr_pipe),
    ('poly', PolynomialFeatures(degree = 1, include_bias=False)),
    ('scale', StandardScaler()),
  ]
)
#
#
#
#
#
#
#
#| echo: true
tree = Pipeline(
    [
        ("preprocessing", preprocessing),
        ("tree", DecisionTreeRegressor(random_state=1)),
    ],
)
#
#
#
#
#
#
#
#
#
#
#
tree
#
#
#
#
#
#
#| echo: true
tree.fit(X_train, y_train)
root_mean_squared_error(y_train, tree.predict(X_train)).round(4)
#
#
#
#
#
#| echo: true
#| code-fold: true
#| code-summary: "Measuring generalization performance with CV"
tree_rmse = -cross_val_score(
    tree,
    X_train,
    y_train,
    scoring="neg_root_mean_squared_error",
    cv=10,
)
pd.Series(tree_rmse).describe().iloc[:3]
```
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#| echo: true
forest_reg = Pipeline(
    [
        ("preprocessing", preprocessing),
        ("random_forest", RandomForestRegressor(n_jobs=-1, random_state=1)),
    ]
)
#
#
#
#
#
#
#
#
#
#| echo: true
forest_rmse = -cross_val_score(
    forest_reg,
    X_train,
    y_train,
    scoring="neg_root_mean_squared_error",
    cv=10,
)
pd.Series(forest_rmse).describe().iloc[:3]
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#| echo: true
param_distribs = {
    "preprocessing__geo__n_clusters": np.random.randint(low=1, high=2),
    "random_forest__max_features": randint(low=2, high=5),
    "random_forest__min_samples_leaf": randint(low=1, high=5)
}
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
