import pandas as pd
import numpy as np
from stock_prediction.params import company_list
import pdb
from datetime import datetime, timedelta



def best_stocks(df, sell=True, eq_weight=False) :
    '''This function allows us to select the best 10 stocks in our
    predictions of returns for each day of the tradding experience.
    We can use ut for our predictions and ask for a inequal weight ponderation
    of the stocks in the portfolio, or call it for the 'True' comparaison
    and ask for an equal ponderation. We can also have the best 10 stocks
    to sell or buy or just to buy. '''

    # the df has a 'stocks' features and one column per day of prediction
    # the first column is 'stocks'
    # as we need to for loop on each day and we don't want to know the number
    # of days, we are going to drop 'stocks' in the column list and loop
    day_pred_list = df.columns[1:]
    # we create a list to store portfolio for each day
    ptf_day_list = []

    for days in day_pred_list :

        # we select only the day we want to analyze
        day_search = df[['stocks', days]].copy()
        # if we work with BUY & SELL we need the absolute returns
        if sell :
            # we create a column with the absolute returns
            day_search['abs_returns'] = day_search[days].abs()
            ten_best = day_search.nlargest(10, 'abs_returns')
            # we need to create the weight column depending on eq_weight
            if eq_weight :
                ten_best['weights'] = 0.10
            else :
                ten_best['weights'] = ten_best['abs_returns'] / ten_best['abs_returns'].sum()
            # finally we can drop the abs_returns column
            ten_best.drop(columns='abs_returns', inplace=True)
        else :
            ten_best = day_search.nlargest(10, days)
            if eq_weight:
                ten_best['weights'] = 0.10
            else:
                ten_best['weights'] = ten_best[days] / ten_best[days].sum()

        # we have a df with the list of 10 best stocks
        # their predicted returns for the day
        # the weight in the portfolio
        # we store it in the list
        # we need to have the stocks as index
        ten_best = ten_best.set_index('stocks')
        ptf_day_list.append(ten_best)

    return ptf_day_list

def true_returns(start_date, end_date, dict_hard_data) :
    '''This function wiil allow us to have a df with all the stocks and their
    respective returns for each day of the simulation
    # we will also store a df with all the Cosing prices to be able to make the BUY/SELL
    # in parameters we have the dates start / end of the simulation
    # also the dictionary with all the inputs from yahoo API, output of workflow.data_collection'''

    # we need valid dates if not errors
    # in the future that would be a point of improvement but no time

    # we need to know the dates gap between start date and end date because some of the stocks
    # does not have the date
    # so we make the search on EuroStoxx to know the gap

    # we create a empty dict to store the data
    dict_open_prices = {}
    dict_close_prices = {}
    dict_true_returns = {}

    for comp in company_list :
        # we select the right df in the dictionary in input
        stock_df = dict_hard_data[comp]
        # selection of the rows depending on dates
        # because we are lazy and missing time, we must find a time period with all stocks traded
        #stock_df = stock_df.iloc[stock_df.loc[stock_df['Date'] == start_date].index[0] - 1 : stock_df[stock_df['Date'] == end_date].index[0]]

        stock_df = stock_df.loc[
          stock_df.loc[stock_df['Date'] == start_date].index[0] - 1 :
            stock_df[stock_df['Date'] == end_date].index[0]]


        # we select on row before to be able to compute the return and have the price of yesterday
        # of the first day to make the BUY/SELL
        # we create the pct_change
        stock_df['Returns_true'] = stock_df['Adj Close'].pct_change(1)
        # don't forget that the first row is not part of our simulation

        #**************************************************************
        # for the rest of the code, all is named "CLOSE" , but making the control
        # in the app trading, we cannot buy at the Close yesterday, knowing the
        # close price , so we need to buy at the open price
        # because no time until presentation of the project, we need to change this and
        # keep the rest of the name on the code as CLOSE but it is the OPEN price
        # in fact, I have more time now :) , train for tomorrow : let's try to complete the task
        #**************************************************************

        open_prices = stock_df[['Date', 'Open']].copy()
        close_prices = stock_df[['Date', 'Close']].copy()
        true_returns = stock_df[['Date', 'Returns_true']].copy()
        # we want to rename the columns to be able to keep the right stock
        # when merging
        open_prices = open_prices.rename(columns={'Open': comp})
        close_prices = close_prices.rename(columns= {'Close' : comp})
        true_returns = true_returns.rename(columns={'Returns_true': comp})
        # we drop the first row
        # to do that we need to reset index
        true_returns.reset_index(drop=True, inplace=True)
        true_returns.drop(labels=0, axis='index')
        # we strore in the dictionaries
        dict_open_prices[comp] = open_prices
        dict_close_prices[comp] = close_prices
        dict_true_returns[comp] = true_returns

    # now we have all the df
    # we need to create a df with all days
    # we merge on dates
    # we retrieve the first one to be able to merge other on it
    # to be sure to have all the dates in the time period for all the stocks
    # we should have -1 row to be able to shift by one in case there is a day trading in
    # the first df that is not for oters
    # or create a first column with the dates we expect to trade during the simulation
    # and merge on that and shift if there is missing value
    df_open_prices = dict_open_prices[company_list[0]]
    df_close_prices = dict_close_prices[company_list[0]]
    df_true_returns = dict_true_returns[company_list[0]]
    for comp in company_list[1:] :
        df_open_prices = df_open_prices.merge(dict_open_prices[comp], how='inner', on='Date')
        df_close_prices = df_close_prices.merge(dict_close_prices[comp], how='inner', on='Date')
        # we do the same for the returns true
        df_true_returns = df_true_returns.merge(dict_true_returns[comp], how='inner', on='Date')
    # we also need to change the index and put the date as indexes
    df_open_prices = df_open_prices.set_index('Date')
    df_close_prices = df_close_prices.set_index('Date')
    df_true_returns = df_true_returns.set_index('Date')

    # now we have our  2 df
    return df_open_prices, df_close_prices, df_true_returns


def portfolio(open_price, close_price, true_returns, best_true, best_pred, cash_invest, true=False) :
    '''This function will use the df in inputs to compute the number of stocks we need to trade
    and impact the cash '''

    # we will make that simulation inside a dictionary that will help us to
    # keep only one of each Date / Stock
    # can call per Stocks / Date easily

    #*******************************
    # to improve the application, we should have to iclude the forex
    # when BUY/SELL the stocks because some of them are not traded in EUR
    #******************************

    # we need the list of the dates we make the simulation
    date_list = close_price.index.to_list()

    # we must declare all the dict to be able to access the keys after and fill them
    dict_dates = {x : 0 for x in date_list}
    # now the dict level 1 of the stocks
    # we include all stocks , cash and daily_return
    dict_ext_variables = {
        'cash': dict_dates.copy(),
        'daily_return_pred': dict_dates.copy(),
        'daily_return_true': dict_dates.copy()
    }
    dict_stocks = {x: dict_dates.copy() for x in company_list}
    # now that e have all the dict ready, we can loop on all the days and store the results as lists
    # lists : nb_stocks, return_pred, return_true, Close_price, Close_true, performance_pred, performance_true

    # we need to give the amount to invest to yesterday of first day of simulation
    dict_ext_variables['cash'][date_list[0]] = cash_invest
    # we also need to separate the process of the best_true ptf and the best_pred_ptf
    if true :
        best_ten = best_true
    else :
        best_ten = best_pred

    # noow the loop on the best_ten df in the lists generated by best_stocks function
    for df in best_ten :

        # we need variables to store values of the cash and preformance
        amount_tot_invested = 0
        perf_daily_saved = 0
        perf_daily_pred = 0
        # we retrieve the date of that ptf
        # the df has Date column and weights columns
        ptf_date = df.drop(columns='weights').columns[0]
        # we need to position of the date in the date_list

        position = date_list.index(ptf_date)
        # we loop on the stocks in the index
        for stocks in df.index.to_list() :
            # the return for that stock at that date (true or pred)
            rets = df.loc[stocks, ptf_date]
            # here we need to know the sell / buy
            if rets > 0 :
                direction = 1
            else :
                direction = -1
            # the weight we want for that stock in the ptf
            weight = df.loc[stocks, 'weights']
            # the amount in $ to trade

            amount_to_trade = weight * dict_ext_variables['cash'][date_list[
                position - 1]]

            # here we need to make a condition
            # because if the move between Close price day -1 and Open price of the day is higher
            # than our prediction ---> our expected return is already done and we cannot make it anymore
            # so we don't buy/ sell if it is the case
            # the move overnight
            overnight = (open_price.loc[date_list[position], stocks] / close_price.loc[date_list[position - 1], stocks]) -1

            if abs(overnight) > abs(rets) :
                nb_stocks = 0

            # number of shares we can buy/sell with this amount
            nb_stocks = amount_to_trade // open_price.loc[date_list[position], stocks]
            # the real amount invested in that stock TRADED ON THE OPEN PRICE OF THE DAY
            amount_traded = nb_stocks * open_price.loc[date_list[position],
                                                        stocks]
            # we need to store that amount to be able to apply to the final cash position at the end of the day
            amount_tot_invested += amount_traded
            # we need the amount at the end of the day regarding the price close at the end of the day
            amount_end_of_day = nb_stocks * close_price.loc[date_list[position], stocks]
            # we find the gains (losses???) at the end of the day, for that stock
            # there is a difference if we sell or buy the stock
            # here we have to take in consideration the case where rets is >0 but the stock
            # goes down , and contrary

            # we also need the true return for that stock that day
            rets_true = true_returns.loc[ptf_date, stocks]

            if rets_true > 0 and rets > 0:
                perform_stock = (amount_end_of_day - amount_traded)
            elif rets_true < 0 and rets < 0:
                perform_stock = (amount_end_of_day - amount_traded) * -1
            elif rets_true > 0 and rets < 0:
                perform_stock = (amount_end_of_day - amount_traded) * -1
            elif rets_true < 0 and rets > 0:
                perform_stock = (amount_end_of_day - amount_traded)

            perf_daily_saved += perform_stock


            # we need the amount we would had if the return predicted was right
            # it takes in consideration the weight and the predicted return
            predicted_return_amount = amount_to_trade * rets * direction
            perf_daily_pred += predicted_return_amount
            # we also need the true return for that stock that day
            rets_true = true_returns.loc[ptf_date, stocks]
            # we also need the performance during the day
            day_perf = (close_price.loc[date_list[position],
                                       stocks] / open_price.loc[
                                           date_list[position], stocks]) - 1
            # now we need to store the values in our dicts
            dict_stocks[stocks][ptf_date] = [
                nb_stocks, rets, rets_true, predicted_return_amount,
                perform_stock, overnight, day_perf
            ]

        # until now we made it for all the stocks in the portfolio for that day
        # before going to the next day, we need to store the global values for that day
        dict_ext_variables['daily_return_pred'][ptf_date] = perf_daily_pred
        dict_ext_variables['daily_return_true'][ptf_date] = perf_daily_saved
        dict_ext_variables['cash'][ptf_date] = dict_ext_variables['cash'][
            date_list[position - 1]] + perf_daily_saved
        #pdb.set_trace()
    return dict_ext_variables, dict_stocks
