import os
from os import path
from parametersMF import *
import sys
import pandas as pd

# date now
date_now = time.strftime("%Y-%m")    
if len(sys.argv) > 2:
    TICKER = sys.argv[1]
    LOOKUP_STEP = int(sys.argv[2])
    model_name = f"{date_now}_{TICKER}-{shuffle_str}-{scale_str}-{split_by_date_str}-{LOSS}-{OPTIMIZER}-{CELL.__name__}-seq-{N_STEPS}-step-{LOOKUP_STEP}-layers-{N_LAYERS}-units-{UNITS}-activation-{ACTIVATION}"
    if BIDIRECTIONAL:
        model_name += "-b"
    ticker_data_filename = os.path.join("data", f"{TICKER}_{date_now}.csv")

if not path.exists(os.path.join("results", model_name) + ".h5"):
    from stock_predictionMF import create_model, load_data
    from tensorflow.keras.layers import LSTM
    from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
    import tensorflow as tf
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

    # create these folders if they does not exist
    if not os.path.isdir("results"):
        os.mkdir("results")

    if not os.path.isdir("logs"):
        os.mkdir("logs")

    if not os.path.isdir("data"):
        os.mkdir("data")

    # load the data
    data = load_data(TICKER, n_steps=N_STEPS, scale=SCALE, split_by_date=SPLIT_BY_DATE, shuffle=SHUFFLE, lookup_step=LOOKUP_STEP, test_size=TEST_SIZE, feature_columns=FEATURE_COLUMNS, ma_periods=MA_PERIODS)
    
    # save the dataframe
    data["df"].to_csv(ticker_data_filename)
    
    # construct the model
    model = create_model(sequence_length=N_STEPS, n_features=len(FEATURE_COLUMNS), loss=LOSS, units=UNITS, cell=CELL, n_layers=N_LAYERS,
                        dropout=DROPOUT, optimizer=OPTIMIZER, bidirectional=BIDIRECTIONAL, activation=ACTIVATION)
    
    # some tensorflow callbacks
    checkpointer = ModelCheckpoint(os.path.join("results", model_name + ".h5"), save_weights_only=True, save_best_only=True, verbose=1)
    tensorboard = TensorBoard(log_dir=os.path.join("logs", model_name))
    
    history = model.fit(data["X_train"], data["y_train"],
                        batch_size=BATCH_SIZE,
                        epochs=EPOCHS,
                        validation_data=(data["X_test"], data["y_test"]),
                        callbacks=[checkpointer, tensorboard],
                        verbose=1)
    
    model.save(os.path.join("results", model_name) + ".h5")