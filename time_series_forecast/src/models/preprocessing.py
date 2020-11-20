import warnings
from collections import defaultdict
import pandas as pd
from pandas.core.common import SettingWithCopyWarning

warnings.filterwarnings(action="ignore", category=SettingWithCopyWarning)

import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import OneHotEncoder

from src.models.automation_models import optimize_models
from src.models.conf_intervals import pred_intervals_rforest, pred_intervals_gbr
from src.models.feature_selector import select_features
from src.models.prophet_modelling import evaluate_prophet
from src import config_file


def combine_results_by_month(data, data_to_combine):
    for idx, row in data.iterrows():
        data.loc[idx, 'monthly_values'] = data_to_combine[(data_to_combine['month'] == idx.month) &
                                                          (data_to_combine['year'] == idx.year)]['Daily Sales'].iloc[0]
    return data


def check_weekends(date):
    if date.weekday() < 5:
        return True
    else:
        return False


def train_predict_process(data, test_data, X_train, y_train, model_dict, dict_results_by_day, final_variables,
                          start_date, end_date, model_name):
    df, test_df = data.copy(), test_data.copy()
    if hasattr(model_dict['model'](), 'n_jobs'):
        clf = model_dict['model'](**model_dict['parameters'], n_jobs=-1)
    else:
        clf = model_dict['model'](**model_dict['parameters'])
    clf.fit(X_train, y_train)

    sub_df = df[(df.index >= pred_month) & (df.index < start_date)]['Daily Sales']
    filtered_test_df = test_df[(test_df.index >= start_date) & (test_df.index <= end_date)][final_variables]
    if model_name == 'GBR':
        upper_model, lower_model = pred_intervals_gbr(X_train, y_train, model['parameters'], percentile=0.95)
        results_day_df = predictor_and_ic_monthly(filtered_test_df, start_date, end_date, clf, 'Daily Sales',
                                                  name=model_name, upper_model=upper_model, lower_model=lower_model)
    else:
        results_day_df = predictor_and_ic_monthly(filtered_test_df, start_date, end_date, clf, 'Daily Sales',
                                                  name=model_name)
    if check_weekends(start_date):  # If False means is a weekend
        dict_results_by_day = accumulate_results(sub_df, dict_results_by_day, results_day_df, model_name=model_name,
                                                 start_date=start_date)
    return dict_results_by_day


def shift_procedure(data, max_shifts=15, response_variable='Daily Sales'):
    """
    Procedure to generate shifts on the response column. In order to mantain the structure first, the full range of
    dates must be defined.
    :param data: pandas DataFrame
    :param max_shifts: int. Maximum number of shifts to generate
    :param response_variable: str. Name of the response variable
    :return: pandas DataFrame containing the shifted columns.
    """
    df = data.copy()
    df_full_period = pd.DataFrame(index=pd.date_range(start=min(df.index), end=max(df.index), freq='D'))
    df_full_period = df_full_period.join(df)
    df_full_period.fillna(-1, inplace=True)  # Fill nans with -1 to avoid dropping true 0
    df_full_shifted = data_column_shift(df_full_period, response_variable, shift_fi=max_shifts, shift_ini=1, shift_by=1,
                                        dropnan=True)
    df_full_shifted = df_full_shifted[df_full_shifted[response_variable] != -1]
    df_full_shifted.replace(to_replace=-1, value=0, inplace=True)

    return df_full_shifted


def encode_column(data, column_to_encode):
    df = data.copy()
    enc = OneHotEncoder()
    enc.fit(df[column_to_encode].values.reshape(-1, 1))
    transformed_array = pd.DataFrame(enc.transform(df[column_to_encode].values.reshape(-1, 1)).toarray(),
                                     index=df.index, columns=[f"{column_to_encode}_{item}"
                                                              for item in enc.categories_[0]])
    result_df = df.join(transformed_array, how="left")
    return result_df, enc


def get_previous_weeks_value(data, column_to_get, week_num=3):
    df = data.copy()
    last_week_dict = dict()
    block_days = [7 * item for item in range(1, week_num + 1)]
    series_df = pd.DataFrame(index=data.index, columns=[f"mean_{day}_previous_{column_to_get}" for day in block_days])
    for idx, row in df.iterrows():
        for i in range(len(block_days)):
            last_week_dict[f"mean_{block_days[i]}_previous_{column_to_get}"] = pd.concat(
                [df[df.index == (idx - timedelta(days=item))][column_to_get] for item in block_days[:i + 1]], axis=0)
        for day_num in last_week_dict:
            if not last_week_dict[day_num].empty:
                series_df.loc[idx, day_num] = last_week_dict[day_num].mean()
            else:
                series_df.loc[idx, day_num] = np.NaN

    series_df = series_df.astype({column: float for column in series_df.columns})
    series_df = backguard_fill_rows(series_df)
    return series_df


def backguard_fill_rows(data):
    df = data.copy()
    for column in df.columns:
        nan_rows = df[df[column].isna()]
        for idx, row in nan_rows.iterrows():
            df.loc[df.index == idx, column] = row[nan_rows.columns != column].mean()

    return df


def generate_range_days(start, end):
    range_days = pd.DataFrame(index=pd.date_range(start=start, end=end,
                                                  freq='D'), columns=['upper_bd', 'lower_bd', 'prediction'])
    return range_days


def predictor_and_ic_monthly(data, predict_day, last_date_in_month, model, response_variable, name, **kwargs):
    df = data.copy()
    results_update = dict()
    index = 1
    results_df = generate_range_days(predict_day, last_date_in_month)
    for idx in results_df.index:
        row = df[df.index == idx]
        if not row.empty:
            row.update(pd.DataFrame(results_update, index=[idx]))
            prediction = model.predict(row.values)
            if name == "RandomForest":
                err_down, err_up = pred_intervals_rforest(model, row, percentile=95)
            elif name == 'GBR':
                if all(kwargs.get(item, False) for item in ['upper_model', 'lower_model']):
                    upper_model, lower_model = kwargs.get('upper_model'), kwargs.get('lower_model')
                    err_down, err_up = lower_model.predict(row.values.reshape(1, -1))[0], \
                                       upper_model.predict(row.values.reshape(1, -1))[0]
                else:
                    raise ValueError("upper and lower models must be provided")
            else:
                print("Model not accepted. Exiting")
                raise NotImplemented
            results_df.loc[idx, 'prediction'] = prediction[0]
            results_df.loc[idx, 'upper_bd'] = err_up
            results_df.loc[idx, 'lower_bd'] = err_down
            results_update[f"{response_variable}_lag_{index}"] = prediction[0]
        else:
            results_df.loc[idx, 'prediction'] = 0  # Non products for the day in the daily data
            results_df.loc[idx, 'upper_bd'] = 0
            results_df.loc[idx, 'lower_bd'] = 0
            results_update[f"{response_variable}_lag_{index}"] = 0
        index += 1
    return results_df


def data_column_shift(data, columns, shift_fi, shift_ini=1, shift_by=1, dropnan=False):
    """
    Shifts columns from shift ini to shift fi by shift by.
    :param data: pandas DataFrame
    :param columns: str or list containing variables from the dataset to shift
    :param shift_ini: int. Initial value default 1
    :param shift_by: int. Step default 1
    :param shift_fi: int
    :param dropnan: bool default True. Take care if more than one of this operations has to be performed together when erasing NA's.
    :return: pandas DataFrame
    """
    cols, names = list(), list()
    df = data.copy()
    if isinstance(columns, str):
        columns = [columns]
    for item in columns:
        cols.extend(
            df[item].shift(i) for i in range(shift_ini, shift_fi + 1, shift_by))  # +1 because python goes from 0 to n-1
        names.extend(f'{item}_lag_{i}' for i in range(shift_ini, shift_fi + 1, shift_by))
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # Combine original dataframe with new created dataframe by index
    df_resulting = df.merge(agg, left_index=True, right_index=True)
    # drop rows with NaN values
    if dropnan:
        df_resulting.dropna(inplace=True)
    return df_resulting


def accumulate_results(data, dict_results_by_day, results_day_df, model_name, start_date):
    df = data.copy()
    expected_value = df.sum() + results_day_df['prediction'].sum()
    upper_bd, lower_bd = df.sum() + results_day_df['upper_bd'].sum(), \
                         df.sum() + results_day_df['lower_bd'].sum()
    dict_results_by_day[model_name][start_date] = {'prediction': expected_value, 'upper_bound': upper_bd,
                                                   'lower_bound': lower_bd}
    print(f"Upper bd {upper_bd}, Lower bd {lower_bd} and Value {expected_value}. Model {model_name}")

    return dict_results_by_day


def generate_features(data):
    df = data.copy()
    # Based on the results obtained from the prophet procedure, let's generate the relevant variables
    df['weekday'], df['monthday'] = df.index.dayofweek, df.index.day
    # Join results from prophet procedure
    df = df.join(prophet_df.loc[:, ['trend', 'monthly', 'weekly']], how="left")
    # Get the value of the previous week 7 days before
    previous_weeks_df = get_previous_weeks_value(df, 'Daily Sales', week_num=3)
    df = df.join(previous_weeks_df, how="left")
    # Try encoder for the weekday
    df, _ = encode_column(df, 'weekday')
    # Finally drop nan values generated in the process
    df.dropna(inplace=True)

    return df


prediction_test_ranges = {
    '2017-11-01': {'start': '2017-11-15', 'end': '2017-11-22'},
    '2017-12-01': {'start': '2017-12-15', 'end': '2017-12-22'},
    '2018-01-01': {'start': '2018-01-15', 'end': '2018-01-25'},
    '2018-02-01': {'start': '2018-02-15', 'end': '2018-02-23'},
}

prediction_validation_ranges = {
    '2017-07-01': {'start': '2017-07-17', 'end': '2017-07-31'},
    '2017-08-01': {'start': '2017-08-15', 'end': '2017-08-31'},
    '2017-09-01': {'start': '2017-09-18', 'end': '2017-09-28'},
    '2017-10-01': {'start': '2017-10-16', 'end': '2017-10-31'},
}

end_date = "2017-11-14"  # For splitting train and test
filter_variables = True  # Apply variable filter
model_features = 'ExtraTreesRegressor'  # Type of model to filter features
# Select if validation or test process
validation_process = False
validation_date = "2017-07-14"
prediction_ranges = prediction_validation_ranges if validation_process else prediction_test_ranges

# 1st step generate lags of 15 days over the daily sales
mydateparser = lambda x: pd.datetime.strptime(x, "%d.%m.%Y")
daily_sales = pd.read_excel(config_file.data_path / 'Exercise - Daily Sales - FOR CANDIDATE-SENT - SHORT.xlsx',
                            sheet_name='Daily Sales', usecols=['Posting Date', 'Daily Sales'],
                            parse_dates=['Posting Date'], date_parser=mydateparser)

# READ Holidays
calendar = pd.read_excel(config_file.data_path / 'Exercise - Working Days calendar - FOR CANDIDATE-SENT - SHORT.xlsx',
                         sheet_name='Calendar', usecols=['Date', 'Country 1'],
                         parse_dates=['Date'], skiprows=4)

# Get prophet model and evalaute the dataset available
holidays = pd.DataFrame({'holiday': 'calendar', 'ds': calendar[calendar['Country 1'] == 0]['Date']})
prophet_df = evaluate_prophet(daily_sales, holidays=holidays, end_date=end_date, plot=True)

# Set indexes
daily_sales = daily_sales.set_index('Posting Date', drop=True)
calendar = calendar.set_index('Date', drop=True)
prophet_df = prophet_df.set_index('ds', drop=True)

# Generate shifted data frame
shifted_df = shift_procedure(daily_sales, max_shifts=15, response_variable='Daily Sales')

# Based on the results obtained from the prophet procedure, let's generate the relevant variables
shifted_df = generate_features(shifted_df)

train_df, test_df = shifted_df[shifted_df.index <= end_date], shifted_df[shifted_df.index > end_date]

# Important features Todo pass to a function
variables_to_use = train_df.columns[train_df.columns != 'Daily Sales'].tolist()
if filter_variables:
    variables_index = select_features(train_df, model_features, variables_to_use, 'Daily Sales',
                                      path_file=config_file.visualizations_path / f"feature_importance_{model_features}")
    # Get the names of the variables returnes by the select_features and keep it
    final_variables = train_df[variables_to_use].columns[variables_index].tolist()
    print(f"Vars to use by the feature selector are\n: {final_variables}")
else:
    final_variables = variables_to_use
    print(f"Vars to use by the feature selector are\n: {final_variables}")

if validation_process:
    print("Validation process")
    # Generating accumulated results
    monthly_sales = daily_sales.groupby([daily_sales.index.year, daily_sales.index.month])['Daily Sales'].sum()
    monthly_sales.index.set_names(['year', 'month'], inplace=True)
    monthly_sales = monthly_sales.reset_index()
    train_df, validation_df = train_df[train_df.index <= validation_date], train_df[train_df.index > validation_date]

# GENERATE X and y (for train and test)
X_train, y_train = train_df.loc[:, final_variables].values, train_df['Daily Sales'].values
# X_test, y_test = test_df.loc[:, final_variables].values, test_df['Daily Sales'].values

# Optimize model parameters by using the hyperopt package.
model_configurations = optimize_models(X_train, y_train)
# Based on the evaluations made previously we decide to use a RandomForest approach so from here the model will be that.

dict_results_by_day = defaultdict(dict)
for pred_month in prediction_ranges:
    start_date, end_date = datetime.strptime(prediction_ranges[pred_month]['start'], "%Y-%m-%d"), \
                           datetime.strptime(prediction_ranges[pred_month]['end'], "%Y-%m-%d")
    # Start iterating by day and finally append the last seen day in the X_train array
    while start_date <= end_date:
        print(f"Predicting results for date {start_date}")
        for model_name in model_configurations:
            model = model_configurations[model_name]
            if validation_process:
                print("Running process for validation purposes")
                dict_results_by_day = train_predict_process(shifted_df, validation_df, X_train, y_train, model,
                                                            dict_results_by_day, final_variables, start_date,
                                                            end_date, model_name)
            else:
                print("Running process for testing purposes")
                dict_results_by_day = train_predict_process(shifted_df, test_df, X_train, y_train, model,
                                                            dict_results_by_day, final_variables, start_date,
                                                            end_date, model_name)
        # Move the window by adding the new observed value into the Train DataSet
        if validation_process:
            X_train = np.append(X_train, validation_df[validation_df.index == start_date][final_variables].values,
                                axis=0)
            y_train = np.append(y_train, validation_df[validation_df.index == start_date]['Daily Sales'].values)
        else:
            X_train = np.append(X_train, test_df[test_df.index == start_date][final_variables].values, axis=0)
            y_train = np.append(y_train, test_df[test_df.index == start_date]['Daily Sales'].values)
        start_date += timedelta(days=1)

# Generate the final DataFrame by model and save it in csv format
for name in dict_results_by_day:
    result_df = pd.DataFrame.from_dict(dict_results_by_day[name], orient='index')
    if validation_process:
        # Append the monthly value obtained before
        result_df = combine_results_by_month(result_df, monthly_sales)
        result_df.to_csv(config_file.data_results_path / f"monthly_results_model_{name}_validation_filter_vars_"
                                                         f"{filter_variables}_{model_features}.csv")
    else:
        result_df.to_csv(config_file.data_results_path / f"monthly_results_model_{name}.csv")
