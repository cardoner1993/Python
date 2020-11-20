from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LassoCV
from xgboost import XGBRegressor
import pandas as pd
import matplotlib.pyplot as plt


def select_features(data, model_tree_name, variables_to_use, response_variable, path_file=None):
    print("Running select from model_ensemble feature selector")
    models = {
        'XGBRegressor': XGBRegressor(),
        'ExtraTreesRegressor': ExtraTreesRegressor(),
        'RandomForestRegressor': RandomForestRegressor(),
        'Lasso': LassoCV(normalize=True, cv=5)
    }
    df = data.copy()
    X_train, y_train = df[variables_to_use], df[response_variable].values
    model = models[model_tree_name]
    model.fit(X_train, y_train)

    selection = SelectFromModel(model, threshold='median', prefit=True)
    feature_idx = selection.get_support()

    if model_tree_name == 'Lasso':
        important_features = pd.Series(model.coef_, index=data[variables_to_use].columns)
    else:
        important_features = pd.Series(selection.estimator.feature_importances_[feature_idx],
                                       index=data[variables_to_use].columns[feature_idx])
    if path_file is not None:
        plot_importance(important_features, model_name=f"feature_selector_{model_tree_name}", path_file=path_file)

    print(f"Important features are {data[variables_to_use].columns[feature_idx]}")

    return feature_idx


def plot_importance(coefs, model_name, path_file):
    coefs.sort_values(inplace=True)
    coefs.plot(kind="barh")
    plt.title(f"Feature importance using {model_name}")
    # Save the figure
    plt.tight_layout()
    plt.savefig(path_file)
    # plt.show()
    plt.clf()
