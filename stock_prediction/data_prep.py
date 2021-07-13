import pandas as pd
import numpy as np
from math import sqrt
import os

from stock_prediction.params import company_list, company_dict


class Data_Prep :

    def __init__(self, name, period):
        self.name = name
        self.period = period

    def find_csv_path(self) :
        '''this function gives us the right csv name file for a specific company'''
        # we retrieve the name of the file in the dict
        csv_file = company_dict[self.name]
        # we want the path of the python file
        we_are = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
        # we build the path of the csv file
        path = we_are + '/raw_data/' + csv_file + '.csv'

        return path

    def load_csv(self) :
        '''laod the csv file to pandas dataframe'''
        data  = pd.read_csv(self.find_csv_path())
        return data

    def data_prep(self) :
        '''Function that make the data preparation for analysis'''
        # first we retrieve the df
        data = self.load_csv()

        # to be able to know the columns we use when df contains several stocks
        # we put the code of the company in each column name
        col_name = company_dict[self.name]

        # we create the column "RETURN" on "Adj Close"
        # why ? Because no impact on dividends and stock splits
        data[f'Return_{col_name}'] = data['Adj Close'].pct_change(1)
        # we create the feature "LOG RETURN" to test which one is working better
        data[f'Log_Return_{col_name}'] = np.log(data["Adj Close"] / data["Adj Close"].shift())
        # we create the feature "HIGH-LOW"
        data[f'High-Low_{col_name}'] = (data['High'] - data['Low']) / data['Low']
        # we create the feature "HIGH-CLOSE" --> difference between closing price and higher price
        data[f'High-Close_{col_name}'] = (data['High'] - data['Close']) / data['Close']
        # same for the difference between the lowest point and the closing price
        # we compute it as a positive value
        data[f'Low-Close_{col_name}'] = ((data['Close'] - data['Low']) / data['Low'])
        # we create the feature "daily evolution of volume", day by day
        data[f'Volume-Change_{col_name}'] = data['Volume'].pct_change(1)
        # we create the feature "volume difference to the mean"
        # we compute the mean of daily volumes in the time period of the analysis
        # then find the difference for each day
        data[f'Period_Volum_{col_name}'] = data['Volume'] / data['Volume'].rolling(self.period).mean() - 1
        # finally volatility
        # one annual vl-olatility, computed on 252 days
        #data[f'Annual_Vol_{col_name}'] = data[f'Return_{col_name}'].rolling(252).std() * sqrt(252)
        # another volatility if we work on a pecific time period
        # or if we want to change that parameter
        data[f'Period_Vol_{col_name}'] = data[f'Return_{col_name}'].rolling(
            self.period).std() * sqrt(252)

        # we can remove the columns we used to compute the new features
        # Open, High, Low, Close, Adj Close

        # we will delete the columns directly rather than select the ones we want
        # like that we don't need to name each time with the company name
        del data['Open']
        del data['High']
        del data['Low']
        del data['Close']
        del data['Adj Close']
        del data['Volume']

        # we call the function to add the index features EuroStoxx 50
        # the function creates the return for the index, volume, and Relative return
        data = self.exo_stoxx50(data)
        # finally we remove the rows with NaN (because volatility calculation)
        # and reset the index
        data = data.drop(index=range(0,self.period))
        data = data.reset_index(drop=True)

        # we return a df with 4 years of prices
        return data

    def select_features(self, df, Return=True, Log_Return=False, High_Low=True, High_Close=True, Low_Close=True,
                        Volume_Change=True, Period_Volum=True, Annual_Vol=True,
                        Period_Vol=True, Return_Index=True, Volum_Index=True, Relative_Return=True) :
        '''Function to be able to remove easily features'''

        # *******************
        # if period < 252 , don't use Annual_vol
        #********************

        col_name = company_dict[self.name]
        # we retrieve our dataframe prepared
        data = df
        if Return == False :
            del data[f'Return_{col_name}']
        if Log_Return == False:
            del data[f'Log_Return_{col_name}']
        if High_Low==False:
            del data[f'High-Low_{col_name}']
        if High_Close==False:
            del data[f'High-Close_{col_name}']
        if Low_Close == False:
            del data[f'Low-Close_{col_name}']
        if Volume_Change == False:
            del data[f'Volume-Change_{col_name}']
        if Period_Volum == False:
            del data[f'Period_Volum_{col_name}']
        # if Annual_Vol == False:
        #     del data[f'Annual_Vol_{col_name}']
        if Period_Vol == False:
            del data[f'Period_Vol_{col_name}']
        if Return_Index == False:
            del data['Return_stoxx_50']
        if Volum_Index == False:
            del data['Period_Volum_stoxx_50']
        if Relative_Return == False:
            del data[f'{col_name}_relatif']

        return data

    def exo_stoxx50(self, df) :
        '''This function will select the indexes we want to be part of the df'''

        # we build the path of the euro stocks csv file
        we_are = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
        path = we_are + '/raw_data/' + '^STOXX50E.csv'
        df_es50 = pd.read_csv(path)
        # we need the code of the company
        col_name = company_dict[self.name]

        # we create the new features in the df es_50
        df_es50['Return_stoxx_50'] = df_es50['Close'].pct_change(1)
        df_es50['Period_Volum_stoxx_50'] = df_es50['Volume'] / df_es50['Volume'].rolling(self.period).mean() - 1
        # we select only the 2 columns we need
        df_es50 = df_es50[['Date', 'Return_stoxx_50', 'Period_Volum_stoxx_50']]
        # we merge on Date
        df = df.merge(df_es50, how='left', on='Date')

        # as we have more Date in the Stock analysed and in the index, we fill the NaN
        df.fillna(value=0.0, inplace=True)

        # then we create the feature Return Relative
        df[f'{col_name}_relatif'] = df[f'Return_{col_name}'] - df['Return_stoxx_50']

        return df

    def Price_Rebase(self, df) :
        '''This function allows us to rebase 100 at the beginning of our time period
        and follow only the return and be able to compare it with exogenous features
        COLUMNS parameter is a list of features to rebase
        This function needs to be used once the final dataframe is ready to modelling'''

        # we retrieve our df
        data = df.copy()
        # to avoid errors on the index we reset_index the df
        data = data.reset_index(drop=True)

        # we select only the columns with "Return" on the name of the columns
        list_col_name = []
        for col in data.columns :
            if col.find("Return") != -1 :
                list_col_name.append(col)

        # we create a df with only the features to rebase
        data_rebased = data[list_col_name].copy()

        # we create the new columns rebased
        for col in list_col_name :
            data_rebased.loc[0, f'{col}_R'] = 100
            # we make a for loop to apply the return to the base 100
            for i in range(1, len(data_rebased)) :
                data_rebased.loc[i, f'{col}_R'] = data_rebased.loc[i-1, f'{col}_R'] * (data_rebased.loc[i, col] + 1)

        # we now want to delete the columns with the returns
        # and keep only the new rebased time series
        # to do that we delete them from data and data_rebased df
        # it cannot del all in one line so we need a for loop
        for col in list_col_name :
            del data_rebased[col]
            del data[col]

        # then we want to merge the rebased df to the data df
        data = pd.concat([data, data_rebased], axis=1)

        return data
