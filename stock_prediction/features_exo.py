import pandas as pd
import numpy as np
import os

def exo_selection(df, sp500=True, gold=True, eurusd=True, nasdaq=True, crude=True, vix=True) :
    '''This function will select the indexes we want to be part of the df
    for the modelisation.
    We need to True/False the indexes in the parameters'''

    # we want the path of the python file
    we_are = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
    path = we_are + '/raw_data/'
    # we need a list to store the column name  to be able to access easily when rebasing
    exo_col_name = []
    # we need to store the df in a list to know at the end which df we need to merge
    exo_df_list = []

    # we load the files we select and create the feature Return
    if sp500 :
        # we build the path of the csv file
        path_of = path + 'S&P500.csv'
        # we load the csv to pd.dtaframe
        data_sp500 = pd.read_csv(path_of)
        # now we want the return of the Adj Close price
        data_sp500['Return_S&P500'] = data_sp500['Close/Last'].pct_change(1)
        # the "Date" column does not have the same format
        # we need to transform it
        data_sp500['Date'] = data_sp500['Date'].map(
            lambda x: f'{x[-4:]}-{x[:2]}-{x[3:5]}')
        # then as we only need the return (in case keep other features) we just keep this column and the Date
        data_sp500 = data_sp500[['Date', 'Return_S&P500']]
        # we store the name of the column
        exo_col_name.append('Return_S&P500')
        # we store the df in the list to merge it later
        exo_df_list.append(data_sp500)
    if gold :
        # we build the path of the csv file
        path_of = path + 'GC=F.csv'
        # we load the csv to pd.dtaframe
        data_gold = pd.read_csv(path_of)
        # now we want the return of the Adj Close price
        data_gold['Return_Gold'] = data_gold['Close'].pct_change(1)
        # then as we only need the return (in case keep other features) we just keep this column and the Date
        data_gold = data_gold[['Date', 'Return_Gold']]
        exo_col_name.append('Return_Gold')
        # we store the df in the list to merge it later
        exo_df_list.append(data_gold)
    if eurusd :
        # we build the path of the csv file
        path_of = path + 'EURUSD=X.csv'
        # we load the csv to pd.dtaframe
        data_usd = pd.read_csv(path_of)
        # now we want the return of the Adj Close price
        data_usd['Return_Usd'] = data_usd['Close'].pct_change(1)
        # then as we only need the return (in case keep other features) we just keep this column and the Date
        data_usd = data_usd[['Date', 'Return_Usd']]
        exo_col_name.append('Return_Usd')
        exo_df_list.append(data_usd)
    if nasdaq:
        # we build the path of the csv file
        path_of = path + '^IXIC.csv'
        # we load the csv to pd.dtaframe
        data_nasdaq = pd.read_csv(path_of)
        # now we want the return of the Adj Close price
        data_nasdaq['Return_Nasdaq'] = data_nasdaq['Adj Close'].pct_change(1)
        # then as we only need the return (in case keep other features) we just keep this column and the Date
        data_nasdaq = data_nasdaq[['Date', 'Return_Nasdaq']]
        exo_col_name.append('Return_Nasdaq')
        exo_df_list.append(data_nasdaq)
    if crude:
        # we build the path of the csv file
        path_of = path + 'CL=F.csv'
        # we load the csv to pd.dtaframe
        data_crude = pd.read_csv(path_of)
        # now we want the return of the Adj Close price
        data_crude['Return_Crude'] = data_crude['Close'].pct_change(1)
        # then as we only need the return (in case keep other features) we just keep this column and the Date
        data_crude = data_crude[['Date', 'Return_Crude']]
        exo_col_name.append('Return_Crude')
        exo_df_list.append(data_crude)
    if vix:
        # we build the path of the csv file
        path_of = path + '^VIX.csv'
        # we load the csv to pd.dtaframe
        data_vix = pd.read_csv(path_of)
        # for the VIX, index of volatility of US markets, we don't want thye return
        # we mill have to keep that index un-rebased
        data_vix['Vix_No_Rebase'] = data_vix['Close'] / 100
        data_vix = data_vix[['Date', 'Vix_No_Rebase']]
        exo_df_list.append(data_vix)

    # now that we have all the df we need to merge them on Date to keep only one column
    # for date and one column for each index return
    # we merge it on the DataFrame returned by Data_prep in data_prep.py

    # first we need to merge the first df of the list on the dataframe data_prep
    for dataframe in exo_df_list :
        df = df.merge(dataframe, how='left', on='Date')

    # here we need to fill NaN to 0 for all Retuns columns
    # but for vix, we need to value of the row -1
    # vix_return column is not in the "exo_col_name" because no need to rebase later

    dict_to_fill = {column: 0.0 for column in exo_col_name}
    df.fillna(value=dict_to_fill, inplace=True)
    df.fillna(method='ffill', inplace=True)

    # we now have the dataframe ready for the modelling
    # this function returns the dataframe

    return df
