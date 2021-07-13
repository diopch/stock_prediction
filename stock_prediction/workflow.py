#---API related

import pandas as pd
import yfinance as yf
from datetime import date, timedelta, datetime
from stock_prediction.data_prep_api import Data_Prep_Api
from stock_prediction.features_exo_api import exo_selection_api
from stock_prediction.arima import arima_multi_day
from stock_prediction.tradding_app import best_stocks, true_returns, portfolio

from stock_prediction.params import company_dict, company_list
import pdb

def data_collection(start_date, period=20) :
    '''This function will load all the yahoo finance data and store them in dictionaries
    We will create different dictionaries for the later uses in the modeling functions
    dict for hard_data : 1 return
    dict for transformed data : 2 return
    df for EuroStoxx 50 : 3 return
    '''

    #************************************************************
    # FIND THE REAL START DATE THAT ALLOWS US TO LOAD ENOUGH DAYS TO MAKE THE MODELLING
    # THINK ABOUT THE 20 DAYS DROP
    start_date_delta = str((datetime.strptime(start_date, '%Y-%m-%d') -
                            timedelta(days=120)).date())
    #************************************************************

    # we want a dictionary to store the yahoo files as they come
    # we will be able to use them when doint the trading application
    # we will also be able to use them when doing the extra exo features
    dict_hard_data = {}
    # we need a dictionary with the df fed with the classical features
    dict_prep_data = {}
    # we use the dictionary where we store the company names and related code in params

    #***********************************************************************************
    # we need to make the work for our index and only load it once and add it to each df
    # we load euro stoxx 50 from yfinance
    if start_date=='max':
        df_es50 = yf.download("^STOXX50E", period="max")
    else :
        df_es50 = yf.download("^STOXX50E",
                              start=str(start_date_delta),
                              end=str(date.today()))
    # we need to reset the index because Date is in index and we want it as features
    df_es50.reset_index(inplace=True)
    # we create the new features in the df es_50
    df_es50['Return_stoxx_50'] = df_es50['Close'].pct_change(1)
    df_es50['Period_Volum_stoxx_50'] = df_es50['Volume'] / df_es50[
        'Volume'].rolling(period).mean() - 1
    # we prepare a df ready to be merged
    es_50_to_merge = df_es50[['Date', 'Return_stoxx_50', 'Period_Volum_stoxx_50']].copy()
    #************************************************************************************
    # we need to load the exogenous features from yahoo finance
    #We use the function in features_exo_api
    # that returns the list of the exogenous indexes and a list with the dataframe with the returns
    # the list of the indexes is full at that time, we will remove them after if needed
    exo_feat = ['eurusd', 'sp500', 'gold', 'nasdaq', 'crude', 'vix']
    exo_col_name, exo_df_list = exo_selection_api(exo_feat, start_date_delta)

    for comp in company_list:

        # we instatiate a class Dat_Prep
        stock = Data_Prep_Api(comp, 20)
        hard_data = stock.load_data(start_date_delta)
        # we need to reset index to have Date as features
        hard_data.reset_index(drop=False, inplace=True)
        # we drop na for the hard_df we will have clean df later
        hard_data.dropna(axis='index', how='any', inplace=True)
        # we want a df with the data from yahoo transformed
        prep_data = stock.data_prep_api(hard_data.copy())
        # we have now the classic df
        # we need to add the information from eurostoxx50
        prep_data = prep_data.merge(es_50_to_merge, how='left', on='Date')
        # we compute the relative return
        prep_data[f'{company_dict[comp]}_relatif'] = prep_data[f'Return_{company_dict[comp]}'] - \
          prep_data['Return_stoxx_50']
        # now we want to merge the exogenous features on the df of the stock
        for exo in exo_df_list :
            prep_data = prep_data.merge(exo, how='left', on='Date')
        # now we have our df ready but we need to fill the possible NaN because
        # US makets are not opened at the same time as Europe
        # we add at the list the columns for EuroStoxx 50
        exo_col_name.append('Return_stoxx_50')
        exo_col_name.append(f'{company_dict[comp]}_relatif')
        exo_col_name.append('Period_Volum_stoxx_50')
        # we create the dict to tell how to fill the possible NaN
        # for the columns with returns
        dict_to_fill = {column: 0.0 for column in exo_col_name}
        prep_data.fillna(value=dict_to_fill, inplace=True)
        # then for vix that needs the previous value
        prep_data.fillna(method='ffill', inplace=True)

        # finally we drop the first rows used to compute stats
        prep_data = prep_data.drop(index=range(0, period))
        # and reset the indexes
        prep_data.reset_index(drop=True, inplace=True)

        #**********************************************
        # we also need to convert the date in DateTime format
        # to string to be able to work with, merge and name columns with
        hard_data['Date'] = hard_data['Date'].apply(lambda x: x.strftime('%Y-%m-%d'))
        prep_data['Date'] = prep_data['Date'].apply(lambda x: x.strftime('%Y-%m-%d'))
        #*********************************************

        # we have now our dataframe for that stock ready, we store it in the dict
        dict_hard_data[comp] = hard_data
        dict_prep_data[comp] = prep_data

    # then we return the 2 dict and 1 df
    return dict_hard_data, dict_prep_data, df_es50

def call_arima(start_date, dict_prep_data, alpha=0.05) :
    ''' This function call the data_collection for each company in the list
    Then call the arima function and returns a dictionary with all df results from arima
    '''
    arima_df = {}

    # we need to trasform the date to datetime
    # and find the number of days from today to start_date
    today = datetime.today()
    days = today - datetime.strptime(start_date, '%Y-%m-%d')
    days = days.days

    # we make a loop with company name that retrieve the df modified
    # and give it to armai function
    for comp in company_list :
        df_stock = dict_prep_data[comp]
        df_arima = arima_multi_day(comp, days, df_stock, alpha=alpha)
        arima_df[comp] = df_arima

    return arima_df

def arima_to_app(start_date, end_date, dict_arima, true=False) :
    '''This funtion processes the results from arima and create the df
    we need to run the trading application '''

    #list to store
    predict_arima = []
    # we loop on the dictionary containing all the df results from the arima model
    for comp in company_list :

        # we need the df in dict_arima corresponding to the company name
        comp_df = dict_arima[comp]
        # first we need to select the rows in the data frame corresponding to the
        # dates of our simulation
        # we retrieve the indexes that are the dates, on a list format
        list_index = comp_df.index.to_list()
        # then take the rows on the full df
        # we need to control if the start_date and end_date are in the list
        # if one of them are not in the list, we take the the full from 0 (for start)
        # and until the end (for end)
        # then when we concat, we will have one or more stocks that give to the df_final
        # too many columns and we will select only the columns from start to end

        if start_date not in list_index :
            start_index = 0
        else :
            start_index = list_index.index(start_date)
        if end_date not in list_index :
            end_index = len(list_index)
        else:
            end_index = list_index.index(end_date)

        sim_df = comp_df.iloc[start_index : end_index + 1]
        # now we want to create a df with the stock as a row and in columns the dates
        #the values are in 'perf_pred columns
        # fisrt we need to retrieve the new list of dates for the simulation
        list_index_sim = sim_df.index.to_list()

        # here we need to specify if we want the best true of the best predicted
        if true :
            df_pred = pd.DataFrame([sim_df['perf_true'].to_list()],
                                   index=[comp],
                                   columns=list_index_sim)
        else :
            df_pred = pd.DataFrame([sim_df['perf_pred'].to_list()],
                                   index=[comp],
                                   columns=list_index_sim)
        # we store it in the list
        predict_arima.append(df_pred)

    # now we need to concatenate all the df
    # we want all the dates represented, if one stock does not have pred
    # for one specific day, we fill NaN with 0
    final_pred = pd.concat(predict_arima, axis=0, join='outer')
    final_pred.fillna(value=0.0, inplace=True)

    # now we need to select only the columns of the selected dates
    # in case the dates asked were not in the list_index, we have to many columns
    # fucking dates
    list_dates_final = final_pred.columns.to_list()
    #pdb.set_trace()
    start_final = list_dates_final.index(start_date)
    end_final = list_dates_final.index(end_date)

    final_pred = final_pred.iloc[:, start_final : end_final + 1]

    # for the app trading we need a column 'stocks' so we need to reset_index and rename the column
    final_pred = final_pred.reset_index()
    final_pred = final_pred.rename(columns={'index': 'stocks'})

    return final_pred

def run_all(start_date, end_date, amount) :
    ''' this function is the one to simulate our trading app
    See Notebook "workflow_test" to have the decomposition '''

    dict_hard_data, dict_prep_data, df_es50 = data_collection(start_date, 20)
    arima_df = call_arima(start_date, dict_prep_data, alpha=0.05)
    final_pred = arima_to_app(start_date, end_date, arima_df, true=False)
    final_true = arima_to_app(start_date, end_date, arima_df, true=True)
    df_open_prices, df_close_prices, df_true_returns = true_returns(
        start_date, end_date, dict_hard_data)
    best_pred = best_stocks(final_pred, sell=True, eq_weight=False)
    best_true = best_stocks(final_true, sell=False, eq_weight=True)
    portfolio_pred = portfolio(df_open_prices, df_close_prices,
                               df_true_returns,
                               best_true,
                               best_pred,
                               amount,
                               true=False)
    portfolio_true = portfolio(df_open_prices, df_close_prices,
                               df_true_returns,
                               best_true,
                               best_pred,
                               amount,
                               true=True)

    return portfolio_pred, portfolio_true
