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

import numpy as np
import pandas as pd
import random

# set seed, so we can get the same results after rerunning several times
np.random.seed(314)
tf.random.set_seed(314)
random.seed(314)

def shuffle_in_unison(a, b):
    # shuffle two arrays in the same way
    state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(state)
    np.random.shuffle(b)

def load_data(TICKER, n_steps=50, scale=True, shuffle=True, lookup_step=1, split_by_date=True,
                test_size=0.2, feature_columns=['adjclose', 'volume', 'open', 'high', 'low']):
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
        df = si.get_data(TICKER, start_date = "01/01/2017")
    elif isinstance(TICKER, pd.DataFrame):
        # already loaded, use it directly
        df = TICKER
    else:
        raise TypeError("ticker can be either a str or a `pd.DataFrame` instances")
    
    # add technical indicators
    # Create Simple Moving Averages
    df['sma5'] = df['adjclose'].rolling(window=5,min_periods=1).mean()
    df['sma20'] = df['adjclose'].rolling(window=20,min_periods=1).mean()
    df['sma50'] = df['adjclose'].rolling(window=50,min_periods=1).mean()
    df['sma100'] = df['adjclose'].rolling(window=100,min_periods=1).mean()
    df['sma200'] = df['adjclose'].rolling(window=200,min_periods=1).mean()
    # Create delta of SMAs
    df['dsma5'] = df['sma5'].diff().fillna(0).astype(float)
    df['dsma20'] = df['sma20'].diff().fillna(0).astype(float)
    df['dsma50'] = df['sma50'].diff().fillna(0).astype(float)
    df['dsma100'] = df['sma100'].diff().fillna(0).astype(float)
    df['dsma200'] = df['sma200'].diff().fillna(0).astype(float)
    # Create acc of SMAs
    df['asma5'] = df['dsma5'].diff().fillna(0).astype(float)
    df['asma20'] = df['dsma20'].diff().fillna(0).astype(float)
    df['asma50'] = df['dsma50'].diff().fillna(0).astype(float)
    df['asma100'] = df['dsma100'].diff().fillna(0).astype(float)
    df['asma200'] = df['dsma200'].diff().fillna(0).astype(float)
    # Create prediction of SMAs
    df['psma5'] = df['sma5'] + df['dsma5'] * lookup_step + df['asma5'] * lookup_step
    df['psma20'] = df['sma20'] + df['dsma20'] * lookup_step + df['asma20'] * lookup_step
    df['psma50'] = df['sma50'] + df['dsma50'] * lookup_step + df['asma50'] * lookup_step
    df['psma100'] = df['sma100'] + df['dsma100'] * lookup_step + df['asma100'] * lookup_step
    df['psma200'] = df['sma200'] + df['dsma200'] * lookup_step + df['asma200'] * lookup_step
    
    # Create MACD
    df['26ema'] = df['adjclose'].ewm(span=26).mean()
    df['12ema'] =  df['adjclose'].ewm(span=12).mean()
    df['9ema'] =  df['adjclose'].ewm(span=9).mean()
    df['MACD'] = (df['12ema']-df['26ema'])
    
    # Create Bollinger Bands
    df['20sd'] = df['adjclose'].rolling(window=20,min_periods=1).std()
    #df.loc[df['20sd'].isnull(),'20sd'] = df['adjclose']
    df['20sd'] = df['20sd'].fillna(0)
    df['upper_band'] = (df['sma20'] + (df['20sd']*2)) - df['adjclose']
    df['lower_band'] = df['adjclose'] - (df['sma20'] - (df['20sd']*2))
    
    # Create Exponential moving average
    df['ema'] = df['adjclose'].ewm(com=0.5).mean()
    # Create Momentum
    df['dprice'] = df['adjclose'].diff().fillna(0).astype(float) 
    df['dvolume'] = df['volume'].diff().fillna(0).astype(float)
    df['momentum'] = (df['dprice'] * df['dvolume'])

    # add date as a column
    if "date" not in df.columns:
        df["date"] = df.index
        
    # this will contain all the elements we want to return from this function
    result = {}
    # we will also return the original dataframe itself
    result['df'] = df.copy()

    # make sure that the passed feature_columns exist in the dataframe
    for col in feature_columns:
        assert col in df.columns, f"'{col}' does not exist in the dataframe."

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
    last_sequence = np.array(df[feature_columns].tail(lookup_step))
    
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
        result["X_train"], result["X_test"], result["y_train"], result["y_test"] = train_test_split(X, y, 
                                                                                test_size=test_size, shuffle=shuffle)

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