from stock_prediction.features_exo import exo_selection
from stock_prediction.data_prep_api import Data_Prep_Api
import numpy as np
import pandas as pd
from statsmodels.tsa.arima_model import ARIMA
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_percentage_error as MAPE

from stock_prediction.params import company_dict, dict_max_train, exo_dict


def arima_multi_day(name, days, df_stocks, alpha=0.05):
    '''This function compute the ARIMA model for a specific stock
    on a period of time.
    It returns a df with predictions, true values, confidence interval
    confidence interval = 1 - alpha
    and the value of the day before to be able to compute the returns
    in the df, features taht we will need later to improve our application
    return prediction, return true, return low confidence, return high confidence
    and dates to be able to merge with other features
    '''

    # we need the number of days we need to train the model
    # days in parameters are the number of days we want with predictions
    # we retrieve the best max_train for the selected company
    best_max_train = dict_max_train[name]
    global_length = best_max_train + days

    # we instantiate the Data_Prep class to create the df
    #stock = Data_Prep(name, best_max_train)
    # with data_prep function we add the features
    #*********************************************
    # WE MUST MODIFY THE data_prep FUNCTION we_are
    #*********************************************
    #data_global = stock.data_prep()
    # with function select_features we select the best exo features
    best_exo_features = exo_dict[name]
    # after several tests, we found that the best exo features for all stocks
    # were High_Close, and Low_Close # if more tests bring us to specific features
    # we will have to make a if for the stocks concerned
    prep_api = Data_Prep_Api(name,period=20)
    data_exo = prep_api.select_features_api(df_stocks,
                                       Return=True,
                                       Log_Return=False,
                                       High_Low=False,
                                       High_Close=True,
                                       Low_Close=True,
                                       Volume_Change=False,
                                       Period_Volum=False,
                                       Period_Vol=False,
                                       Return_Index=False,
                                       Volum_Index=False,
                                       Relative_Return=False)
    # we select the rows needed
    data_exo = data_exo[- global_length : ]

    # we need the code of the company to be able to re-create the name of the rebased return
    code_name = company_dict[name]

    # we rebase 100 the return
    data_exo = prep_api.Price_Rebase_api(data_exo)

    # we create the y_endogenous and y_exogenous
    y_endo = data_exo[f'Return_{code_name}_R']
    y_exo = np.array(data_exo[best_exo_features])

    # now we prepare the for loop that will train the model on max_train
    # and gives us a prediction price (rebase) for each day of the time period

    # we need lists to store the results of the loop
    list_y_pred = []
    #baseline = []
    real_value = []
    y_before = []
    y_conf_low = []
    y_conf_high = []
    std_conf = []

    # we fix our parameters for the ARIMA
    order = (0,1,0)
    # number of splits to cover the full time period
    splits = len(y_endo) - best_max_train

    # with function TimeSeriesSplit we create the indexes
    folds = TimeSeriesSplit(n_splits=splits,
                            max_train_size=best_max_train,
                            test_size=1)

    # now the for loop to compute the ARIMA model on each day
    for (train_idx, test_idx) in folds.split(y_endo) :

        # we retrieve the real data in y
        #corresponding of the indexes splited
        y_train = y_endo[train_idx]
        y_exo_train = y_exo[train_idx]
        y_test = y_endo[test_idx]
        y_exo_test = y_exo[test_idx]
        y_true = y_endo[test_idx]
        #base = y_train.iloc[-1]

        # fit our model on y_train of the split n°X
        model = ARIMA(endog=y_train, exog=y_exo_train, order=order).fit()

        # we find our y_pred for this slice on that part of the TS
        y_pred, std_pred, conf = model.forecast(steps=len(y_test), exog=y_exo_test, alpha=alpha)
        #pdb.set_trace()
        # we store the low price of confidence
        y_conf_low.append(conf[0][0])
        # high price of confidence
        y_conf_high.append(conf[0][1])
        # we store y_pred
        list_y_pred.append(y_pred[0])
        # we store the std of the conf interval, see if it helps us
        std_conf.append(std_pred[0])

        # we store the value for basescore
        #baseline.append(base)

        # and the rel value to compare
        real_value.append(y_true.values[0])
        y_before.append(y_train.iloc[-1])

    # once we have all values we can compute MAPE and basescore
    # for purpose of application, we don't need them,
    # used before to compare models
    #base_score = MAPE(baseline, real_value)
    #mape_metric = MAPE(list_y_pred, real_value)

    # we now need to store the results found to be able to work on them
    multi_days_results = np.array([y_before, list_y_pred, y_conf_low, y_conf_high, real_value])
    multi_days_results_df = pd.DataFrame({'yesterday' : y_before,
                                      'prediction' : list_y_pred,
                                      'conf_low' : y_conf_low,
                                      'conf_high' : y_conf_high,
                                      'true' : real_value,
                                      'conf_std' : std_conf,
                                      'Date' : data_exo['Date'][-splits :]})

    # now we have a numpy array that will helps to make calculus on features
    # and the df that will store the resulted columns

    # pred / before -1
    perf_pred = ((multi_days_results[1, :] / multi_days_results[0, :]) - 1)
    # true / before - 1
    perf_true = ((multi_days_results[4, :] / multi_days_results[0, :]) - 1)

    # direction de pred
    # that can gives us tha accuracy of UP/DOWN
    # if needed in the future we can add it to the df from here
    #dir_pred = perf_pred > 0
    # direction de true
    #dir_true = perf_true > 0
    # accurate direction
    #dir_acc = dir_pred == dir_true

    # perf of Low conf
    perf_low = (multi_days_results[2, :] / multi_days_results[0, :] - 1)
    # perf high conf
    perf_high = (multi_days_results[3, :] / multi_days_results[0, :] - 1)

    # we try to analyze the confidence
    # if we cannot make a new model on our results,
    # we will have to analyze the conf interval and try improving results
    #conf_ana = perf_high - (-perf_low)
    #dir_conf_low = perf_low > 0
    #dir_conf_high = perf_high > 0
    #conf_confirm = dir_conf_low == dir_conf_high

    # we store the results in the global df
    multi_days_results_df['perf_pred'] = perf_pred
    multi_days_results_df['perf_true'] = perf_true
    multi_days_results_df['perf_low'] = perf_low
    multi_days_results_df['perf_high'] = perf_high

    multi_days_results_df = multi_days_results_df.set_index('Date')

    return multi_days_results_df
