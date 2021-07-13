import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from numpy.random import default_rng

from stock_prediction.data_prep_api import Data_Prep_Api
from stock_prediction.tradding_app import best_stocks, true_returns, portfolio
from stock_prediction.tradding_app import true_returns, portfolio, best_stocks
from stock_prediction.workflow import data_collection, call_arima, arima_to_app, run_all
from stock_prediction.tradding_app import true_returns, portfolio, best_stocks
from PIL import Image
import base64


#---Logo sidebar
image1 = Image.open('img/le_wagon_small.png')
image2 = Image.open('img/eurostoxx-50.png')
st.sidebar.image(image2, caption='EuroStoxx50', use_column_width=False) #[image1, image2]
# st.image(image1)
st.image(image1, caption='Le Wagon', use_column_width=False)
# st.sidebar.image(image2, caption='EuroStoxx50', use_column_width=False)


#---Set a title
# st.title('Stock return prediction')
st.markdown("<h1 style='text-align: center; color: black;'>Stock return prediction</h1>", unsafe_allow_html=True)
# st.markdown("<h1 style='text-align: center; color: black;'>1 day traiding application</h1>", unsafe_allow_html=True)



# #---Background image
# main_bg = "img/background_presentation.jpg" #background_presentation.jpg, Le_wagon_fond_ecran.png
# main_bg_ext = "jpg"
# st.markdown(
#     f"""
#     <style>
#     .reportview-container {{
#         background: url(data:image/{main_bg_ext};base64,{base64.b64encode(open(main_bg, "rb").read()).decode()})
#     }}
#     </style>
#     """,
#     unsafe_allow_html=True)



#---Select start date and end date for dataframe generation to show prediction for best stocks
st.sidebar.markdown("**Select a traiding period**")

start_date = st.sidebar.date_input(
    "Please, enter start date:",
    datetime.date(2021, 5, 31))

end_date = st.sidebar.date_input(
    "Please, enter end date:",
    datetime.date(2021, 6, 11))

# Select price
st.sidebar.markdown("**Are you ready to invest?**")

invest = st.sidebar.slider('Please, enter the investement amount from 10.000€ to 100.000€ :',10000,100000,50000, 10000)



#---Visualize all stocks table
# st.markdown("""
# **Predictive returns for Euro Stoxx 50**
#             """)
@st.cache
def visualize_stocks():

    #---Retrieve df to visualize
    dict_hard_data, dict_prep_data, df_es50 = data_collection(str(start_date), 20)

    #---True returns df
    open_price, df_close_prices, df_true_returns = true_returns(str(start_date), str(end_date), dict_hard_data)

    # run the arima model
    arima_df = call_arima(str(start_date), dict_prep_data, alpha=0.05)   
    
    #---Pred
    # organize arima result per day and per stocks for the predictive returns (our simulation)
    final_pred = arima_to_app(str(start_date), str(end_date), arima_df, true = False)
    # construct our portfolio (best 10 predicted returns in absolute / BUY and SELL for one day)
    best_pred = best_stocks(final_pred, sell=True, eq_weight=False)
    
    #---True
    # organize arima result per day and per stocks for the predictive returns (our simulation)
    final_true = arima_to_app(str(start_date), str(end_date), arima_df, true = True)
    # construct the best portfolio for the day but only BUY (if market down, must BUY anyway)
    best_true = best_stocks(final_true, sell=False, eq_weight=True)

    # the final function
    portfolio_pred, portfolio_true = run_all(str(start_date), str(end_date), invest)

    return arima_df, final_pred, best_pred, best_true, dict_hard_data, dict_prep_data, df_es50, df_close_prices, df_true_returns, portfolio_pred, portfolio_true

arima_df, final_pred, best_pred, best_true, dict_hard_data, dict_prep_data, df_es50, df_close_prices, df_true_returns, portfolio_pred, portfolio_true = visualize_stocks()

#---Select day to visualize 10 best companies to invest
list_days = pd.date_range(start = str(start_date), end = str(end_date)).strftime("%Y-%m-%d").to_list()
st.sidebar.markdown("**Let's visualize Top 10 stocks**")
select_day = st.sidebar.selectbox('Please, select a day for prediction:', list_days)
day = list_days.index(select_day)

#---Prediction for best stocks
# st.markdown("<h3 style='text-align: center; color: blue;'>Top 10 stocks prediction</h3>", unsafe_allow_html=True)
best_pred_day = best_pred[day].drop(columns = 'weights')
best_pred_day = best_pred_day*100
best_pred_day.columns=[f'% Return pred on {select_day}']
# st.write(best_pred_day)

#---True return for best stocks
# st.markdown("<h3 style='text-align: center; color: black;'>Top 10 stocks in reality</h3>", unsafe_allow_html=True)
best_true_day = best_true[day]



#---True return (buying on new day open price)
best_pred_stocks = best_pred[day].index.to_list()
return_true = df_true_returns.T.loc[best_pred_stocks][[select_day]]*100
return_true.columns=[f'% Return on {select_day}']

#---Retrieve predicted_return_amount for Top pred and perform_stock for Top true, + Change_overnight, % Open-Close
keys = best_pred[day].index
predicted_return_amount = [x[select_day][3] for x in list(map(portfolio_pred[1].get, keys))]
perform_stock = [x[select_day][4] for x in list(map(portfolio_pred[1].get, keys))]

change_overnight = [x[select_day][5] for x in list(map(portfolio_pred[1].get, keys))]
open_close = [x[select_day][6] for x in list(map(portfolio_pred[1].get, keys))]

#---Concatenate init df of returns with amounts
best_pred_day['Expected amount, €'] = pd.Series(predicted_return_amount, index = best_pred_day.index)
return_true['Real amount, €'] = pd.Series(perform_stock, index = return_true.index)

best_pred_day['Change overnight'] = pd.Series(change_overnight, index = best_pred_day.index)*100
return_true['% Open-Close'] = pd.Series(open_close, index = return_true.index)*100


#-------------------------------------------------color red negative values
def color_set(val):
    """
    Takes a scalar and returns a string with
    the css property `'color: red'` for negative
    strings, black otherwise.
    """
    color = 'red' if val < 0 else 'black'
    return 'color: %s' % color

#---1.Visualize classic way
# # Prediction
# st.markdown("<h3 style='text-align: center; color: red;'>Top 10 stocks Prediction</h3>", unsafe_allow_html=True)
# #------------------------------------------------st.write(best_pred_day)
# #highlite
# st.dataframe(best_pred_day.style.highlight_max(axis=0))

# # True
# st.markdown("<h3 style='text-align: center; color: red;'>Top 10 stocks</h3>", unsafe_allow_html=True)
# #------------------------------------------------st.write(return_true)
# st.dataframe(return_true.style.highlight_max(axis=0))

#---2.Make columns to visualize our prediction and real top 10 stocks
cols_title = st.beta_columns(2)
cols_title[0].markdown("<h3 style='text-align: center; color: steelblue;'>Top 10 stocks Prediction</h3>", unsafe_allow_html=True)
cols_title[1].markdown("<h3 style='text-align: center; color: darkorange;'>Top 10 stocks</h3>", unsafe_allow_html=True)

cols = st.beta_columns(2)
cols[0].dataframe(best_pred_day.style.applymap(color_set)) # highlight_max(axis=0)
cols[1].dataframe(return_true.style.applymap(color_set)) #best_true_day



#---Visualize price Expected vs Real
st.markdown("<h3 style='text-align: center; color: black;'>Top 10 stocks 1 day return Expected vs Real</h3>", unsafe_allow_html=True)
pred_amount = best_pred_day[[f'% Return pred on {select_day}']]
true_amount = return_true[[f'% Return on {select_day}']]
df_concat = pd.concat([pred_amount, true_amount], axis=1)
st.bar_chart(df_concat)

#---Table & Plot performance
df_performance = pd.DataFrame.from_dict(portfolio_pred[0])

st.markdown("<h3 style='text-align: center; color: black;'>Performance evolution</h3>", unsafe_allow_html=True)
#-----------------------------------------------st.write(df_performance)
st.table(df_performance)

st.markdown("<h3 style='text-align: center; color: black;'>Daily return Expected vs Real</h3>", unsafe_allow_html=True)
st.line_chart(df_performance[['daily_return_pred', 'daily_return_true']])



if st.button('Thank you for your attention !🎈🎈🎈 '):
    st.balloons()










