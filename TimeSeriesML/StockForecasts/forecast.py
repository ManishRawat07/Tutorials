from parameters import *
import sys

# date now
date_now = time.strftime("%Y-%m")
if len(sys.argv) > 2:
    TICKER = sys.argv[1]
    LOOKUP_STEP = int(sys.argv[2])
    model_name = f"{date_now}_{TICKER}-{shuffle_str}-{scale_str}-{split_by_date_str}-{LOSS}-{OPTIMIZER}-{CELL.__name__}-seq-{N_STEPS}-step-{LOOKUP_STEP}-layers-{N_LAYERS}-units-{UNITS}-activation-{ACTIVATION}"
    if BIDIRECTIONAL:
        model_name += "-b"
        
import os
from os import path
if not path.exists(os.path.join("results", model_name) + ".h5"):
    print("Model not found: .\\results\\"+model_name)
    quit()

from stock_prediction import create_model, load_data, np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

def predict(model, data):
    # retrieve the last sequence from data
    last_sequence = data["last_sequence"][-N_STEPS:]
    # expand dimension
    last_sequence = np.expand_dims(last_sequence, axis=0)
    # get the prediction (scaled from 0 to 1)
    prediction = model.predict(last_sequence)
    # get the price (by inverting the scaling)
    if SCALE:
        predicted_price = data["column_scaler"]["adjclose"].inverse_transform(prediction)[0][0]
    else:
        predicted_price = prediction[0][0]
    return predicted_price

# load the data
data = load_data(TICKER, N_STEPS, scale=SCALE, split_by_date=SPLIT_BY_DATE, 
                shuffle=SHUFFLE, lookup_step=LOOKUP_STEP, test_size=TEST_SIZE, 
                feature_columns=FEATURE_COLUMNS)

# construct the model
model = create_model(N_STEPS, len(FEATURE_COLUMNS), loss=LOSS, units=UNITS, cell=CELL, n_layers=N_LAYERS, dropout=DROPOUT, optimizer=OPTIMIZER, bidirectional=BIDIRECTIONAL, activation=ACTIVATION)

model_path = os.path.join("results", model_name) + ".h5"
model.load_weights(model_path)

# evaluate the model
#mse, mae = model.evaluate(data["X_test"], data["y_test"], verbose=0)
# calculate the mean absolute error (inverse scaling)
#mean_absolute_error = data["column_scaler"]["adjclose"].inverse_transform([[mae]])[0][0]
#print("Mean Absolute Error:", mean_absolute_error)
# predict the future price
future_price = predict(model, data)
print(f"Future $ price after {LOOKUP_STEP} days is {future_price:.2f}")
