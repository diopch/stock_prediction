import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras import Sequential, layers, models
from tensorflow.keras.layers.experimental.preprocessing import Normalization
from tensorflow.keras.utils import normalize

from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.metrics import MAPE, MAE, MSE
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import regularizers

#---Import moduls from stock-prediction package
from stock_prediction.data_prep import Data_Prep
from stock_prediction.features_exo import exo_selection

#---API related
import os 
from math import sqrt 
import yfinance as yf
from datetime import date, timedelta, datetime
from stock_prediction.data_prep_api import Data_Prep_Api
from stock_prediction.features_exo_api import exo_selection_api

# companies dict with name in key to load and idx in value
company_dict = {
    'asml' : 'ASML.AS',
    'lvmh': 'MC.PA',
    'sap' : 'SAP.DE',
    'linde' : 'LIN',
    'siemens' : 'SIE.DE',
    'total' : 'FP.PA',
    'sanofi' : 'SAN.PA',
    'allianz' : 'ALV.DE', 
    'schneider' : 'SU.PA',
    'iberdrola' : 'IBE.MC',
    'enel' : 'ENEL.MI',
    'air-liquide' : 'AI.PA',
    'basf' : 'BAS.DE',
    'bayer' : 'BAYN.DE',
    'adidas' : 'ADS.DE',
    'airbus' : 'AIR.PA',
    'adyen' : 'ADYEN.AS',
    'deutsche-telecom' : 'DTE.DE',
    'daimler' : 'DAI.DE',
    'bnp' : 'BNP.PA',
    'anheuser-busch' : 'ABI.BR',
    'vinci' : 'DG.PA',
    'banco-santander' : 'SAN.MC',
    'philips' : 'PHIA.AS',
    'kering' : 'KER.PA',
    'deutsche-post' : 'DPW.DE',
    'axa' : 'CS.PA',
    'safran' : 'SAF.PA',
    'danone'  : 'BN.PA',
    'essilor' : 'EL.PA',
    'intensa' : 'ISP.MI',
    'munchener' : 'MUV2.DE',
    'pernod' : 'RI.PA',
    'vonovia' : 'VNA.DE',
    'vw' : 'VOW3.DE',
    'ing' : 'INGA.AS',
    'crh' : 'CRG.IR',
    'industria-diseno' : 'ITX.MC',
    'kone' : 'KNEBV.HE',
    'deutsche-borse' : 'DB1.DE',
    'ahold' : 'AHOG.DE',
    'flutter' : 'FLTR.IR',
    'amadeus' : 'AMS.MC',
    'engie' : 'ENGI.PA',
    'bmw' : 'BMW.DE',
    'vivendi' : 'VIV.PA',
    'eni' : 'ENI.MI',
    'nokia' : 'NOKIA.HE'
}

#---Load API data function for a selected period
def load_data_api(company, start_date='2017-05-31', end_date='2019-04-10', len_=30):

    prep_class = Data_Prep_Api(company, len_)
    df = prep_class.data_prep_api(max=True) # (max=True)
    # prep_class.select_features_api(df, Return = True, Log_Return=False, High_Low=True, High_Close=True, Low_Close=True,
    #                     Volume_Change=False, Period_Volum=True, Annual_Vol=False,
    #                     Period_Vol=True, Return_Index=True, Volum_Index=True, Relative_Return=True)

    prep_class.select_features_api(df, Return = True, Log_Return=False, High_Low=True, High_Close=True, Low_Close=True,
                        Volume_Change=False, Period_Volum=True, Annual_Vol=False,
                        Period_Vol=False, Return_Index=True, Volum_Index=True, Relative_Return=False)

    df = exo_selection_api(df, ["sp500", "eurusd", "crude", "vix"], max=True) # ,max=True) "gold", "nasdaq",

    df = df.sort_values('Date')
    
    start_date_delta = str((datetime.strptime(start_date, '%Y-%m-%d') - timedelta(days=len_)).date()) # replace with extracted len_ from loaded model
    mask = (df['Date'] >= start_date_delta) & (df['Date'] < end_date)
    df = df.loc[mask]
    
    df = df.set_index('Date')
    
    idx = df.columns[0][7:]
    
    return df, idx

#---Preprocessing sequences functions
def train_test_val_split(df, horizon=1, train_threshold=0.6, val_threshold=0.8):
    """This function allows us to split sets chronologically
     in order too avoid data leakage."""

    # here gap=0, if we take a longer horizon it'll matter
    gap = horizon - 1

    # set a train: 60%, test: 20%, val: 20% sizes
    train = int( train_threshold*df.shape[0])
    val = int(val_threshold*df.shape[0])


    df_train = df[:train]
    df_val = df[train:val]
    df_test = df[val+gap:] # gap can matter later
    
    return df_train, df_val, df_test


def shift_sequences(df, idx, length=30, horizon=1):
    
    """This function is able to get as many subsequences for X and a corresponding target y
    Nas possible, taking in account lenght of sequence and horizon."""
    
    last_possible = df.shape[0] - length - horizon
    X=[]
    y=[]
    for start in range(last_possible):
        X.append(df[start: start+length].values)
        y.append(df.iloc[start+length+horizon-1][f'Return_{idx}'])
    
    # random permutation of the sequences to train
    perm = np.random.permutation(last_possible)
    X = np.array(X)[perm, :, :]
    y = np.array(y)[perm]
    return X, y


#--shift_sequences in order
def shift_sequences_pred(df, idx, length=30, horizon=1):
    
    """This function is able to get as many subsequences for X and a corresponding target y
    Nas possible, taking in account lenght of sequence and horizon."""
    
    last_possible = df.shape[0] - length - horizon
    X=[]
    y=[]
    for start in range(last_possible):
        X.append(df[start: start+length].values)
        y.append(df.iloc[start+length+horizon-1][f'Return_{idx}'])

    X = np.array(X)
    y = np.array(y)
    return X, y


#---Train model Pipeline
def train_model(df,
                idx,
                train_threshold=0.6,
                val_threshold=0.8,
                nb_sequences=50,
                len_=30,
                l_rate=0.01,
                momentum=0.9,
                loss='MAE',
                metric=MAE,
                patience=50,
                batch_size=128,
                horizon=1,
                verbose=2):
    """Train model function:
    Allows to split sequences and gets X and y for train, val, test sets;
    Initializes a model with (Normalization), LSTM/GRU/Conv1D and 2 fully connected layers.
    Returns: X_train, y_train, X_test, y_test, model"""
    
#     regul = 0

    if not isinstance(df, list):
        df = [df]

    X_train = []
    y_train = []
    X_val = []
    y_val = []
    X_test = []
    y_test = []
    for nb_df, df_loc in enumerate(df):
        idx_loc = idx[nb_df]
        
        #---Split train, test, val sets
        df_train, df_val, df_test = train_test_val_split(
            df_loc, train_threshold=train_threshold, val_threshold=val_threshold)

        #---Get train, test, val X and y sequences
    #     X_train_loc, y_train_loc = get_X_y(df_train, idx_loc, length=len_, number_of_sequences=nb_sequences)
    #     X_val_loc, y_val_loc = get_X_y(df_val, idx_loc, length=len_, number_of_sequences=nb_sequences) #round(nb_sequences*0.2)
    #     X_test_loc, y_test_loc = get_X_y(df_test, idx_loc, length=len_, number_of_sequences=nb_sequences) #round(nb_sequences*0.2)

        X_train_loc, y_train_loc = shift_sequences(df_train, idx_loc, length=len_)
        X_val_loc, y_val_loc = shift_sequences(df_val, idx_loc, length=len_)
        X_test_loc, y_test_loc = shift_sequences(df_test, idx_loc, length=len_)
        
        X_train.append(X_train_loc)
        y_train.append(y_train_loc)
        
        X_val.append(X_val_loc)
        y_val.append(y_val_loc)
        
        X_test.append(X_test_loc)
        y_test.append(y_test_loc)
    
    # Data concatenation
    X_train = np.concatenate(X_train)
    y_train = np.concatenate(y_train)
    X_val = np.concatenate(X_val)
    y_val = np.concatenate(y_val)
    X_test = np.concatenate(X_test)
    y_test = np.concatenate(y_test)
    
    # Data shuffle (random, different for train / val / test)
    perm = np.random.permutation(len(X_train))
    X_train, y_train = X_train[perm], y_train[perm]
    perm = np.random.permutation(len(X_val))
    X_val, y_val = X_val[perm], y_val[perm]
    perm = np.random.permutation(len(X_test))
    X_test, y_test = X_test[perm], y_test[perm]
    
#     print(np.any(np.isnan(X_train)))
#     print(np.any(np.isnan(X_val)))
#     print(np.any(np.isnan(X_test)))

    normalizer = Normalization()
    normalizer.adapt(X_train)  # , axis=-1, order=1 as minmax

    #---Initialize the model
    model = Sequential()
    model.add(layers.InputLayer(input_shape=(len_, X_train.shape[-1])))
    model.add(normalizer)
    
#     model.add(layers.GRU(50, return_sequences=True, activation='tanh'))
#     model.add(layers.GRU(30, return_sequences=True, activation='tanh'))
#     model.add(layers.GRU(20, return_sequences=False, activation='tanh'))

    model.add(layers.Conv1D(128, 2, activation='relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.MaxPool1D(pool_size=2))
    
    model.add(layers.Conv1D(64, 2, activation='relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.MaxPool1D(pool_size=2))
    
    model.add(layers.Conv1D(32, 2, activation='relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.MaxPool1D(pool_size=2))
    
    model.add(layers.Flatten())

    model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dropout(0.1))
    model.add(layers.Dense(1, activation='linear'))
    
#     model.add(layers.LSTM(50, return_sequences=True, activation='tanh', kernel_regularizer=regularizers.l2(regul)))
#     model.add(layers.LSTM(30, return_sequences=True, activation='tanh', kernel_regularizer=regularizers.l2(regul)))
#     model.add(layers.LSTM(20, return_sequences=False, activation='tanh', kernel_regularizer=regularizers.l2(regul)))

#     model.add(layers.Dense(20, activation='relu', kernel_regularizer=regularizers.l2(regul)))
#     model.add(layers.Dense(1, activation='linear', kernel_regularizer=regularizers.l2(regul)))

    model.compile(
        loss=loss,
        optimizer=Adam(l_rate),  # RMSprop(learning_rate=l_rate, momentum=momentum), 
        metrics=[metric])
    
    print(X_train.shape)
    print(model.summary())

    es = EarlyStopping(monitor='val_loss',
                       patience=patience,
                       restore_best_weights=True)
    
    lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                           patience=round(patience/2))

    history = model.fit(X_train,
                        y_train,
                        validation_data=(X_val, y_val),
                        epochs=1000,
                        batch_size=batch_size,
                        callbacks=[es, lr],
                        verbose=verbose)

    stop_epoch = max(history.epoch)-patience
    val_mae = history.history['val_mean_absolute_error'][stop_epoch]

    print(f"early stopping at {max(history.epoch)} epoch.\nval_mae: {val_mae}")

    return X_train, y_train, X_val, y_val, X_test, y_test, model


if __name__ == '__main__':
    #---Load data for all companies from csv to train
    data = []
    idx = []
    for company in company_dict.keys():
        data_loc, idx_loc = load_data_api(company, start_date='2012-01-01', end_date='2019-10-04', len_=30) # Pre-covid data
        data.append(data_loc)
        idx.append(idx_loc)


    #---Train model on Pre_covid data
    X_train, y_train, X_val, y_val, X_test, y_test, model = train_model(df=data,
                                                                        idx=idx,
                                                                        len_=30, 
                                                                        l_rate=0.01,
                                                                        batch_size=128,
                                                                        patience=50,
                                                                        loss='MAE',
                                                                        metric=MAE,
                                                                        verbose=2)


    #---Save trained model
    savedir = datetime.now().strftime('saved_model_cnn_%Y_%m_%d_%H_%M_%S')
    savepath = os.path.join('saved_models_cnn', savedir)
    os.makedirs(savepath)

    model.save(savepath)


#---Function that does return prediction for a given period for selected companies
def prediction_return_cnn(start='2016-07-05', stop = '2021-06-11'):

    """This function allows to automatically load the data for a given period with an API,
        to select same features, that model was trained with,
        to load trained CNN model
        and to make stock return prediction per company for a given period of time."""

    #---Load trained model
    path = os.path.dirname(os.path.abspath(__file__))
    model = models.load_model(os.path.join(path, 'saved_models_cnn/model_cnn_all_tr_2012_2019_f30_lr0001_p50_/')) #Reaplce with a proper model ex.: model_cnn_vinci_tr_2012_2019_f20

    len_ = model.layers[0].output_shape[1]

    prediction = {}

    for company in company_dict.keys():
        data_to_predict_loc, idx__loc = load_data_api(company, start_date=start, end_date=stop, len_=len_)

        X_test_loc, y_test_loc = shift_sequences_pred(df = data_to_predict_loc, idx = idx__loc, length=len_)
        
        prediction_loc = model.predict(X_test_loc)
        prediction[company] = [prediction_loc.ravel(), data_to_predict_loc.index]

    return prediction