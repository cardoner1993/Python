import numpy as np
from sklearn.ensemble import GradientBoostingRegressor


def pred_intervals_rforest(model, row, percentile=95):
    preds = []
    for pred in model.estimators_:
        preds.append(pred.predict(row.values.reshape(1, -1))[0])
    err_down = np.percentile(preds, (100 - percentile) / 2.)
    err_up = np.percentile(preds, 100 - (100 - percentile) / 2.)
    return err_down, err_up


def pred_intervals_gbr(X_train, y_train, parameters, percentile=0.95):
    # Each model has to be separate
    lower_alpha, upper_alpha = (1 - percentile) / 2., 1 - (1 - percentile) / 2.
    lower_model = GradientBoostingRegressor(loss="quantile",
                                            alpha=lower_alpha, **parameters)
    upper_model = GradientBoostingRegressor(loss="quantile",
                                            alpha=upper_alpha, **parameters)

    # Fit the models
    lower_model.fit(X_train, y_train)
    upper_model.fit(X_train, y_train)

    return upper_model, lower_model

