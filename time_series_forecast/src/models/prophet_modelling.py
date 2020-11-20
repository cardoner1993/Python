import pandas as pd
from src import config_file
from fbprophet import Prophet
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

end_date = "2017-10-31"


def evaluate_prophet(data, holidays, end_date, y_column='Daily Sales', plot=True):
    """
    Generate prophet procedure in order to obtain the components of the time series
    :param data: pandas DataFrame
    :param holidays: pd DataFrame indicating the holiday events
    :param end_date:
    :param y_column: str. Name of the response variable.
    :param plot: bool default True. If plot components and forecast must be shown.
    :return: pandas DataFrame
    """
    df = data.copy()
    if not hasattr(df, "ds"):
        date_col = df.select_dtypes(include=['datetime64']).columns.tolist()[0]
        df.rename(columns={date_col: 'ds', y_column: 'y'}, inplace=True)
    train_sales, test_sales = df[df['ds'] <= end_date], df[df['ds'] > end_date]
    clf = Prophet(holidays=holidays)
    clf.add_seasonality(name='monthly', period=30.5, fourier_order=5)
    clf.fit(train_sales)
    periods = len(pd.date_range(start=min(test_sales['ds']), end=max(test_sales['ds']), freq='D'))
    future = clf.make_future_dataframe(periods=periods, freq='D', include_history=True)
    predictions = clf.predict(future)
    # return predictions[predictions['ds'] <= end_date], predictions[predictions['ds'].isin(test_sales['ds'])]
    if plot:
        # plot forecast and components
        clf.plot_components(predictions)
        plt.show()
        clf.plot(predictions)
        plt.title("Ṕrophet predictions")
        plt.show()
    return predictions


if __name__ == '__main__':

    # Daily sales
    mydateparser = lambda x: pd.datetime.strptime(x, "%d.%m.%Y")
    daily_sales = pd.read_excel(config_file.data_path / 'Exercise - Daily Sales - FOR CANDIDATE-SENT - SHORT.xlsx',
                                sheet_name='Daily Sales', usecols=['Posting Date', 'Daily Sales'],
                                parse_dates=['Posting Date'], date_parser=mydateparser)
    # daily_sales = daily_sales.set_index('Posting Date', drop=True)

    # Monthly values
    # monthly_sales = pd.read_excel(config_file.data_path / 'Exercise - ACT and LO Monthly - FOR CANDIDATE-SENT - SHORT.xlsx',
    #                             sheet_name='Act and LO')


    # Calendar efects
    calendar = pd.read_excel(config_file.data_path / 'Exercise - Working Days calendar - FOR CANDIDATE-SENT - SHORT.xlsx',
                                sheet_name='Calendar', usecols=['Date', 'Country 1'],
                                parse_dates=['Date'], skiprows=4)
    # calendar = calendar.set_index('Date', drop=True)


    # Use fbprophet to detect seasonalities and other effects
    # Reformat the calendar dataframe in orther to match fbprophet requirements

    daily_sales.rename(columns={'Posting Date': 'ds', 'Daily Sales': 'y'}, inplace=True)

    holidays = pd.DataFrame({'holiday': 'calendar', 'ds': calendar[calendar['Country 1'] == 0]['Date']})

    train_sales, test_sales = daily_sales[daily_sales['ds'] <= end_date], daily_sales[daily_sales['ds'] > end_date]

    clf = Prophet(holidays=holidays)
    clf.add_seasonality(name='monthly', period=30.5, fourier_order=5)
    clf.fit(train_sales)
    periods = len(pd.date_range(start=min(test_sales['ds']), end=max(test_sales['ds']), freq='D'))
    future = clf.make_future_dataframe(periods=periods, freq='D', include_history=True)

    predictions = clf.predict(future)
    pred_array = predictions[predictions['ds'].isin(test_sales['ds'])]['yhat'].values
    test_array = test_sales['y'].values

    print(f"MSE is {mean_squared_error(test_array, pred_array)}")

    #plot forecasted values and components
    clf.plot(predictions)
    plt.show()
    clf.plot_components(predictions)
    plt.show()

    # Visualization
    fig = plt.figure(figsize=(6, 6))
    plt.subplot(1, 1, 1)
    plt.plot(test_array, 'b-', label='True values')
    plt.plot(pred_array, 'r-', label='Predicted values')
    plt.legend(loc='upper right')
    fig.tight_layout()
    plt.show()

