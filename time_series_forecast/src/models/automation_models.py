import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, KFold, RepeatedKFold
from sklearn.linear_model import Ridge, Lasso
from hyperopt import tpe, hp, Trials
from hyperopt.fmin import fmin
from functools import partial


seed = 2

models_list = {
    'RandomForest': {'model': RandomForestRegressor,
                     'parameters': {'n_estimators': hp.quniform('n_estimators', 25, 500, 25),
                                    'max_depth': hp.quniform('max_depth', 1, 10, 1),
                                    'n_jobs': -1},
                     'types_parameters': {'n_estimators': int, 'max_depth': int}},
    'GBR': {'model': GradientBoostingRegressor,
            'parameters': {'max_depth': hp.quniform('max_depth', 2, 8, 1),
                           'subsample': hp.uniform('subsample', 0.3, 1.0),
                           'n_estimators': hp.quniform('n_estimators', 25, 500, 25),
                           'learning_rate': 0.05},
            'types_parameters': {'n_estimators': int, 'max_depth': int, 'subsample': float}},
    # 'Lasso': {'model': Lasso, 'parameters': {'alpha': hp.uniform('alpha', 0, 1.0), 'normalize': True},
    #           'types_parameters': {'alpha': float}},
    # 'Ridge': {'model': Ridge, 'parameters': {'alpha': hp.uniform('alpha', 0, 1.0), 'normalize': True},
    #           'types_parameters': {'alpha': float}}
}


def objective(params, train_X, train_y, model, params_types):
    params_type = {key: params_types[key](params[key]) if key in params_types else params[key] for key in params}
    clf = model(**params_type)  # Instantiate the model
    score = cross_val_score(clf, train_X, train_y, scoring='neg_mean_squared_error', cv=KFold()).mean()
    return score


def optimize(trial, objective, params):
    best2 = fmin(fn=objective, space=params, algo=tpe.suggest, trials=trial, max_evals=50,
                 rstate=np.random.RandomState(seed))
    return best2


def optimize_models(X_train, y_train):
    dict_results = dict()
    for model in models_list:
        trial = Trials()
        parameters = models_list[model]['parameters']
        parameter_types = models_list[model]['types_parameters']
        objective_fun_rgr = partial(objective, train_X=X_train, train_y=y_train, model=models_list[model]['model'],
                                    params_types=parameter_types)
        best = optimize(trial, objective_fun_rgr, parameters)
        best_params = {key: parameter_types[key](best[key]) if key in parameter_types else best[key] for
                       key in best}
        clf = models_list[model]['model'](**best_params)
        # Validate results for each model in a cross validation procedure (10 folds, 3 repeats)
        scores = evaluate_model(X_train, y_train, repeats=3, model=clf)
        final_score = np.sqrt(np.mean(abs(scores)))  # SQRT to make it easy for analyzing
        dict_results[model] = {'model': models_list[model]['model'], 'parameters': best_params, 'score': final_score}

    return dict_results


def evaluate_model(X, y, repeats, model):
    # prepare the cross-validation procedure
    cv = RepeatedKFold(n_splits=10, n_repeats=repeats, random_state=1)
    # evaluate model
    scores = cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=cv, n_jobs=-1)
    return scores