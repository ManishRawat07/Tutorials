import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)
import tensorflow.keras.backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from yahoo_fin import stock_info as si
from collections import deque

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import random
import pypyodbc
cnxn = pypyodbc.connect("Driver={SQL Server Native Client 11.0};"
                        "Server=127.0.0.1;"
                        "Database=database;"
                        "uid=username;pwd=password")
                        
# set seed, so we can get the same results after rerunning several times
np.random.seed(314)
tf.random.set_seed(314)
random.seed(314)

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only use the first GPU
  try:
    tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
  except RuntimeError as e:
    # Visible devices must be set at program startup
    print(e)

def shuffle_in_unison(a, b):
    # shuffle two arrays in the same way
    state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(state)
    np.random.shuffle(b)
    
def get_rsi_timeseries(prices, n=1):
    # RSI = 100 - (100 / (1 + RS))
    # where RS = (Wilder-smoothed n-period average of gains / Wilder-smoothed n-period average of -losses)
    # Note that losses above should be positive values
    # Wilder-smoothing = ((previous smoothed avg * (n-1)) + current value to average) / n
    # For the very first "previous smoothed avg" (aka the seed value), we start with a straight average.
    # Therefore, our first RSI value will be for the n+2nd period:
    #     0: first delta is nan
    #     1:
    #     ...
    #     n: lookback period for first Wilder smoothing seed value
    #     n+1: first RSI

    # First, calculate the gain or loss from one price to the next. The first value is nan so replace with 0.
    deltas = (prices-prices.shift(1)).fillna(0)

    # Calculate the straight average seed values.
    # The first delta is always zero, so we will use a slice of the first n deltas starting at 1,
    # and filter only deltas > 0 to get gains and deltas < 0 to get losses
    avg_of_gains = deltas[1:n+1][deltas > 0].sum() / n
    avg_of_losses = -deltas[1:n+1][deltas < 0].sum() / n

    # Set up pd.Series container for RSI values
    rsi_series = pd.Series(0.0, deltas.index)

    # Now calculate RSI using the Wilder smoothing method, starting with n+1 delta.
    up = lambda x: x if x > 0 else 0
    down = lambda x: -x if x < 0 else 0
    i = n+1
    for d in deltas[n+1:]:
        avg_of_gains = ((avg_of_gains * (n-1)) + up(d)) / n
        avg_of_losses = ((avg_of_losses * (n-1)) + down(d)) / n
        if avg_of_losses != 0:
            rs = avg_of_gains / avg_of_losses
            rsi_series[i] = 100 - (100 / (1 + rs))
        else:
            rsi_series[i] = 100
        i += 1

    return rsi_series
    
def tr(data):
    data['previous_close'] = data['close'].shift(1)
    data['high-low'] = abs(data['high'] - data['low'])
    data['high-pc'] = abs(data['high'] - data['previous_close'])
    data['low-pc'] = abs(data['low'] - data['previous_close'])
    tr = data[['high-low', 'high-pc', 'low-pc']].max(axis=1)
    return tr

def atr(data, period):
    data['tr'] = tr(data)
    atr = data['tr'].rolling(period).mean()
    return atr
    
def supertrend(df, period=7, atr_multiplier=2.5):
    hl2 = (df['high'] + df['low']) / 2
    df['atr'] = atr(df, period)
    df['upperband'] = hl2 + (atr_multiplier * df['atr'])
    df['lowerband'] = hl2 - (atr_multiplier * df['atr'])
    df['in_uptrend'] = 1

    for current in range(1, len(df.index)):
        previous = current - 1

        if df['close'][current] > df['upperband'][previous]:
            df['in_uptrend'][current] = 1
        elif df['close'][current] < df['lowerband'][previous]:
            df['in_uptrend'][current] = 0
        else:
            df['in_uptrend'][current] = df['in_uptrend'][previous]

            if df['in_uptrend'][current] and df['lowerband'][current] < df['lowerband'][previous]:
                df['lowerband'][current] = df['lowerband'][previous]

            if not df['in_uptrend'][current] and df['upperband'][current] > df['upperband'][previous]:
                df['upperband'][current] = df['upperband'][previous]
        
    return df

def load_data(TICKER, n_steps=50, scale=True, shuffle=True, lookup_step=1, split_by_date=True,
                test_size=0.2, feature_columns=['adjclose', 'volume'], ma_periods=[5, 20], endDate="12/31/2070"):
    """
    Loads data from Yahoo Finance source, as well as scaling, shuffling, normalizing and splitting.
    Params:
        ticker (str/pd.DataFrame): the ticker you want to load, examples include AAPL, TESL, etc.
        n_steps (int): the historical sequence length (i.e window size) used to predict, default is 50
        scale (bool): whether to scale prices from 0 to 1, default is True
        shuffle (bool): whether to shuffle the data, default is True
        lookup_step (int): the future lookup step to predict, default is 1 (e.g next day)
        test_size (float): ratio for test data, default is 0.2 (20% testing data)
        feature_columns (list): the list of features to use to feed into the model, default is everything grabbed from yahoo_fin
    """
    # see if ticker is already a loaded stock from yahoo finance
    if isinstance(TICKER, str):
        # load it from yahoo_fin library
        # df = si.get_data(TICKER, start_date = "01/01/2020", end_date=endDate)
        query = "SELECT [date_time] as date_time, round([price_open]/100.0, 2) as [open], round([price_high]/100.0, 2) as [high],  round([price_low]/100.0, 2) as [low],  round([price_close]/100.0, 2) as [close],  round([adjClose]/100.0, 2) as [adjclose], [volume], '"+TICKER+"' as ticker  FROM [StockTitan].[st].[daily] where stock_id = (select top(1) stock_id from [st].[stock] where symbol = '"+TICKER+"' and status > 0) and date_time >= getdate() - 821 and date_time <= '"+endDate+"' and volume > 0 and price_close > 0 order by date_time"
        df = pd.read_sql_query(query, cnxn)
        df = df.set_index('date_time')
        df['adjclose'].iloc[-1] = df['close'].iloc[-1]
        #df['adjclose'].iloc[-2] = df['close'].iloc[-2]
        #df['adjclose'].iloc[-3] = df['close'].iloc[-3]
        #df['adjclose'].iloc[-4] = df['close'].iloc[-4]
        #df['adjclose'].iloc[-5] = df['close'].iloc[-5]
        df = df[df.adjclose != 0]

    elif isinstance(TICKER, pd.DataFrame):
        # already loaded, use it directly
        df = TICKER
    else:
        raise TypeError("ticker can be either a str or a `pd.DataFrame` instances")
    
    #Calculate the On Balance Volume
    OBV = []
    OBV.append(0)

    # add technical indicators
    #for n in ma_periods:
        # Create Simple Moving Averages
        #df['sma'+str(n)] = df['adjclose'].rolling(window=n,min_periods=1).mean()
        # Create delta of SMAs
        #df['dsma'+str(n)] = df['sma'+str(n)].diff().fillna(0).astype(float)
        # Create acc of SMAs
        #df['asma'+str(n)] = df['dsma'+str(n)].diff().fillna(0).astype(float)
        # Create prediction of SMAs
        #df['psma'+str(n)] = df['sma'+str(n)] + df['dsma'+str(n)] * lookup_step + df['asma'+str(n)] * lookup_step
        # Create RSI of SMAs
        #df['rsi'+str(n)] = get_rsi_timeseries(df['sma'+str(n)], n)
        # Create EMA
        #df['ema'+str(n)] = df['adjclose'].ewm(span=n).mean()
        ## Create Bollinger Bands
        #df['sd'+str(n)] = df['adjclose'].rolling(window=n,min_periods=1).std()
        #df['sd'+str(n)] = df['sd'+str(n)].fillna(0)
        #df['upper_band'+str(n)] = (df['sma'+str(n)] + (df['sd'+str(n)]*2)) - df['adjclose']
        #df['lower_band'+str(n)] = df['adjclose'] - (df['sma'+str(n)] - (df['sd'+str(n)]*2))
    df['hlc3'] = (df['close'] + df['high'] + df['low']) / 3.0
    df['sma3'] = df['hlc3'].ewm(span=3).mean()
    df['ema1'] = df['adjclose'].ewm(span=7).mean()
    df['ema2'] = df['adjclose'].ewm(span=14).mean()
    df['ema3'] =  df['adjclose'].ewm(span=21).mean()
    df['MACD'] = (df['ema1'] - df['ema3'])
    df['rsi'] = get_rsi_timeseries(df['adjclose'], 14)
    df['SMArsi'] = df['rsi'].ewm(span=14).mean()
    df['drsi'] = df['rsi'] - df['SMArsi']
    #df['MACDS'] = df['MACD'].ewm(span=7).mean()
    #df['MACDD'] = df['MACD'] - df['MACDS']
    
    # Create Bollinger Bands
    df['sma20'] = df['adjclose'].rolling(window=20,min_periods=1).mean()
    #df['sma7'] = df['adjclose'].rolling(window=7,min_periods=1).mean()
    #df['sma14'] = df['adjclose'].rolling(window=14,min_periods=1).mean()
    #df['sma21'] = df['adjclose'].rolling(window=21,min_periods=1).mean()
    #df['sma30'] = df['adjclose'].rolling(window=30,min_periods=1).mean()
    #df['20sd'] = df['adjclose'].rolling(window=20,min_periods=1).std()
    #df['20sd'] = df['20sd'].fillna(0)
    #df['upper_band'] = (df['sma20'] + (df['20sd']*2)) - df['adjclose']
    #df['lower_band'] = df['adjclose'] - (df['sma20'] - (df['20sd']*2))
    df = supertrend(df)
    df['kst'] = df['sma3'] - df['ema1'] + df['sma3'] - df['ema2'] + df['sma3'] - df['ema3']
    df['wema1'] = df['hlc3'] - df['ema1']
    df['wema2'] = df['hlc3'] - df['ema2']
    df['wema3'] = df['hlc3'] - df['ema3']
    
    # Create Momentum
    df['dprice'] = df['adjclose'].diff().fillna(0).astype(float) 
    df['dvolume'] = df['volume'].diff().fillna(0).astype(float)
    df['momentum'] = (df['dprice'] * df['dvolume'])
    
     # On Balance Volume * Price Calcs
    #for i in range(1, len(df.adjclose)):
    #    if df.adjclose[i] > df.adjclose[i-1]: #If the closing price is above the prior close price 
    #        OBV.append(OBV[-1] + (df.volume[i] * df.adjclose[i])) #then: Current OBV = Previous OBV + Current Volume * Price
    #    elif df.adjclose[i] < df.adjclose[i-1]:
    #        OBV.append( OBV[-1] - (df.volume[i] * df.adjclose[i]))
    #    else:
    #        OBV.append(OBV[-1])
    #Store the OBV and OBV EMA into new columns
    #df['OBV'] = OBV        
    #df['OBVFast'] = df['OBV'].rolling(window=7,min_periods=1).mean()
    #df['OBVSlow'] = df['OBV'].rolling(window=65,min_periods=1).mean()
    #df['dOBV'] = df['OBVFast'] - df['OBVSlow'].fillna(0).astype(float)
    df['SMAVol20'] = df['volume'].rolling(window=20,min_periods=1).mean()
    df['volmom'] = df['volume'] -  df['SMAVol20']
    #df['dEMAFS'] = df['7ema'] - df['21ema'].fillna(0).astype(float)
    df['perc1c2']=df['adjclose'].shift(1) / df['adjclose'].shift(2)
    df['percc1']=df['adjclose']/df['adjclose'].shift(1).fillna(0).astype(float)
    df['ROC10']=df['adjclose'] /df['adjclose'].shift(10).fillna(0).astype(float)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df = df.fillna(0)

    # add date as a column
    if "date" not in df.columns:
        df["date"] = df.index

    # make sure that the passed feature_columns exist in the dataframe
    for col in feature_columns:
        assert col in df.columns, f"'{col}' does not exist in the dataframe."
   
    # this will contain all the elements we want to return from this function
    result = {}

    # we will also return the original dataframe itself
    result['df'] = df.copy()

    if scale:
        column_scaler = {}
        # scale the data (prices) from 0 to 1
        for column in feature_columns:
            scaler = preprocessing.MinMaxScaler()
            df[column] = scaler.fit_transform(np.expand_dims(df[column].values, axis=1))
            column_scaler[column] = scaler

        # add the MinMaxScaler instances to the result returned
        result["column_scaler"] = column_scaler

    # add the target column (label) by shifting by `lookup_step`
    df['future'] = df['adjclose'].shift(-lookup_step)

    # last `lookup_step` columns contains NaN in future column
    # get them before droping NaNs
    last_sequence = np.array(df[feature_columns].tail(lookup_step*12))
    
    # drop NaNs
    df.dropna(inplace=True)

    sequence_data = []
    sequences = deque(maxlen=n_steps)

    for entry, target in zip(df[feature_columns + ["date"]].values, df['future'].values):
        sequences.append(entry)
        if len(sequences) == n_steps:
            sequence_data.append([np.array(sequences), target])

    # get the last sequence by appending the last `n_step` sequence with `lookup_step` sequence
    # for instance, if n_steps=50 and lookup_step=10, last_sequence should be of 60 (that is 50+10) length
    # this last_sequence will be used to predict future stock prices that are not available in the dataset
    last_sequence = list([s[:len(feature_columns)] for s in sequences]) + list(last_sequence)
    last_sequence = np.array(last_sequence).astype(np.float32)
    # add to result
    result['last_sequence'] = last_sequence
    
    # construct the X's and y's
    X, y = [], []
    for seq, target in sequence_data:
        X.append(seq)
        y.append(target)

    # convert to numpy arrays
    X = np.array(X)
    y = np.array(y)

    if split_by_date:
        # split the dataset into training & testing sets by date (not randomly splitting)
        train_samples = int((1 - test_size) * len(X))
        result["X_train"] = X[:train_samples]
        result["y_train"] = y[:train_samples]
        result["X_test"]  = X[train_samples:]
        result["y_test"]  = y[train_samples:]
        if shuffle:
            # shuffle the datasets for training (if shuffle parameter is set)
            shuffle_in_unison(result["X_train"], result["y_train"])
            shuffle_in_unison(result["X_test"], result["y_test"])
    else:    
        # split the dataset randomly
        try:
            result["X_train"], result["X_test"], result["y_train"], result["y_test"] = train_test_split(X, y, 
                                                                                test_size=test_size, shuffle=shuffle)
        except:
            return

        # get the list of test set dates
        dates = result["X_test"][:, -1, -1]
        # retrieve test features from the original dataframe
        result["test_df"] = result["df"].loc[dates]
        # remove dates from the training/testing sets & convert to float32
        result["X_train"] = result["X_train"][:, :, :len(feature_columns)].astype(np.float32)
        result["X_test"] = result["X_test"][:, :, :len(feature_columns)].astype(np.float32)


    return result
    
def custom_loss(y_true, y_pred):
    #custom_loss = K.mean(K.sum(K.square(y_true - y_pred)))
    alpha = 100.
    loss = K.switch(K.less(y_true * y_pred, 0), \
        alpha*y_pred**2 - K.sign(y_true)*y_pred + K.abs(y_true), \
        K.abs(y_true - y_pred)
        )
    return K.mean(loss, axis=-1)
    #return custom_loss

def create_model(sequence_length, n_features, units=128, cell=LSTM, n_layers=4, dropout=0.3,
                loss="mean_absolute_error", optimizer="rmsprop", bidirectional=False, activation="sigmoid"):
    model = Sequential()
    for i in range(n_layers):
        if i == 0:
            # first layer
            if bidirectional:
                model.add(Bidirectional(cell(units, return_sequences=True), batch_input_shape=(None, sequence_length, n_features)))
            else:
                model.add(cell(units, return_sequences=True, batch_input_shape=(None, sequence_length, n_features)))
        elif i == n_layers - 1:
            # last layer
            if bidirectional:
                model.add(Bidirectional(cell(units, return_sequences=False)))
            else:
                model.add(cell(units, return_sequences=False))
        else:
            # hidden layers
            if bidirectional:
                model.add(Bidirectional(cell(units, return_sequences=True)))
            else:
                model.add(cell(units, return_sequences=True))
        # add dropout after each layer
        model.add(Dropout(dropout))
    model.add(Dense(1, activation=activation))
    if loss == "custom_loss":
        model.compile(loss=custom_loss, metrics=["mean_absolute_error"], optimizer=optimizer)
    else:
        model.compile(loss=loss, metrics=["mean_absolute_error"], optimizer=optimizer)
    
    return model