from stock_prediction.data_prep import Data_Prep
from stock_prediction.features_exo import exo_selection

import numpy as np
import pandas as pd
from pandas import datetime
from arch import arch_model

def garch(name, days) :
    '''This function returns a df with 'days' days with a prevision of volatility
    for the name stock selected '''

    # we load the datas
    # we instantiate the Data_Prep class to create the df
    # we don't care about the period as we won't use the other features
    stock = Data_Prep(name, 20)
    # with data_prep function we add the features
    #*********************************************
    # WE MUST MODIFY THE data_prep FUNCTION we_are
    #*********************************************
    data_global = stock.data_prep()

    # we want to keep the full time period to train the model
    # we only need the returns and date
    data_global = data_exo = stock.select_features(data_global,
                                                   Return=True,
                                                   Log_Return=False,
                                                   High_Low=False,
                                                   High_Close=False,
                                                   Low_Close=False,
                                                   Volume_Change=False,
                                                   Period_Volum=False,
                                                   Annual_Vol=False,
                                                   Period_Vol=False,
                                                   Return_Index=False,
                                                   Volum_Index=False,
                                                   Relative_Return=False)

    # we select our y
    y_returns = data_global.drop(columns='Date')
    # the model converges better if we *100 the returns
    y_returns = y_returns * 100
    # we create the time period on it we want the prediction
    split = 50
    # list to store results
    rolling_predictions = []

    # for loop to compute the volatility for each day of the dataframe
    for i in range(split, len(y_returns)):
        train = y_returns[i - split:i + split]
        # we instatiate the model on the split days
        # we will see if we change the parameters
        # for the moment we keep the default parameters
        # mean = ‘Constant’, ‘Zero’, ‘LS’, ‘AR’, ‘ARX’, ‘HAR’ and ‘HARX’
        # vol = ‘GARCH’ (default), ‘ARCH’, ‘EGARCH’, ‘FIARCH’ and ‘HARCH’
        # dist = Normal , Students, skewed Students, generalized error
        model_garch = arch_model(
            train,
            vol='Garch',
            p=1,
            q=1,
            dist='Normal',
            mean='Constant'
        )
        # we fit on the last year / rows available
        model_fit = model_garch.fit(disp='off')
        pred = model_fit.forecast(horizon=1)

        # we store the result (variance of residuals)
        rolling_predictions.append(np.sqrt(pred.variance.values[0][0]))

    # we annualize the daily volatility
    rolling_predictions = np.array(rolling_predictions) * np.sqrt(252)

    # we build a df with the dates to be able to merge on dates
    garch_pred = pd.DataFrame({
        'garch_pred': rolling_predictions,
        'Date': data_global['Date'][50:]
    })

    # we want also the volatility variation day by day
    # to see if it helps to predict
    garch_pred['vol_variation'] = garch_pred['garch_pred'].pct_change(1)
    garch_pred.dropna(inplace=True)

    # finally we select the right number of days we want to analyse
    garch_pred = garch_pred.iloc[-days : ]

    return garch_pred
