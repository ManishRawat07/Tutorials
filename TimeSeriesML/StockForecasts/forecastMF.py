from parametersMF import *
import sys
from stock_predictionMF import pd

pd.options.mode.chained_assignment = None  # default='warn'

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

from stock_predictionMF import create_model, load_data, np
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

def predict_gap(model, data, df2, indexVals):
    predicted_prices = []
    for X in range(LOOKUP_STEP+1):
        # retrieve the last sequence from data
        last_sequence = data["last_sequence"][-N_STEPS-X:]
        last_sequence = last_sequence[:N_STEPS]
        # expand dimension
        last_sequence = np.expand_dims(last_sequence, axis=0)
        # get the prediction (scaled from 0 to 1)
        prediction = model.predict(last_sequence)
        # get the price (by inverting the scaling)
        if SCALE:
            predicted_price = data["column_scaler"]["adjclose"].inverse_transform(prediction)[0][0]
        else:
            predicted_price = prediction[0][0]   

        df2.loc[indexVals[X-LOOKUP_STEP],'forecast'] = predicted_price
    return df2

def plot_graph2(test_df, df2):
    """
    This function plots true close price along with predicted close price
    with blue and red colors respectively
    """
    plt.figure(figsize=(18,8))
    plt.title(TICKER+" Stock Price Forecast "+ f"{LOOKUP_STEP}" +" days out", fontsize=16)
    plt.plot(test_df[f'true_adjclose_{LOOKUP_STEP}'].tail(N_STEPS), c='b')
    plt.plot(test_df[f'adjclose_{LOOKUP_STEP}'].tail(N_STEPS), c='r')
    plt.plot(df2['forecast'].tail(LOOKUP_STEP+1), c='r')
    plt.xlabel("Days")
    plt.ylabel("Price")
    plt.legend(["Actual Price", "Predicted Price"])
    forecast_folder = "forecasts"
    if not os.path.isdir(forecast_folder):
        os.mkdir(forecast_folder)
    filename = os.path.join(forecast_folder, TICKER.lower() + "_" + f"{LOOKUP_STEP}" + "_forecast.png")
    plt.savefig(filename)
    #plt.show()    
   
def get_final_df(model, data):
    """
    This function takes the `model` and `data` dict to 
    construct a final dataframe that includes the features along 
    with true and predicted prices of the testing dataset
    """
    # if predicted future price is higher than the current, 
    # then calculate the true future price minus the current price, to get the buy profit
    buy_profit  = lambda current, true_future, pred_future: true_future - current if pred_future > current else 0
    # if the predicted future price is lower than the current price,
    # then subtract the true future price from the current price
    sell_profit = lambda current, true_future, pred_future: current - true_future if pred_future < current else 0
    X_test = data["X_test"]
    y_test = data["y_test"]
    # perform prediction and get prices
    y_pred = model.predict(X_test)
    if SCALE:
        y_test = np.squeeze(data["column_scaler"]["adjclose"].inverse_transform(np.expand_dims(y_test, axis=0)))
        y_pred = np.squeeze(data["column_scaler"]["adjclose"].inverse_transform(y_pred))
    test_df = data["test_df"]
    # add predicted future prices to the dataframe
    test_df[f"adjclose_{LOOKUP_STEP}"] = y_pred
    # add true future prices to the dataframe
    test_df[f"true_adjclose_{LOOKUP_STEP}"] = y_test
    # sort the dataframe by date
    test_df.sort_index(inplace=True)
    final_df = test_df
    # add the buy profit column
    final_df["buy_profit"] = list(map(buy_profit, 
                                    final_df["adjclose"], 
                                    final_df[f"adjclose_{LOOKUP_STEP}"], 
                                    final_df[f"true_adjclose_{LOOKUP_STEP}"])
                                    # since we don't have profit for last sequence, add 0's
                                    )
    # add the sell profit column
    final_df["sell_profit"] = list(map(sell_profit, 
                                    final_df["adjclose"], 
                                    final_df[f"adjclose_{LOOKUP_STEP}"], 
                                    final_df[f"true_adjclose_{LOOKUP_STEP}"])
                                    # since we don't have profit for last sequence, add 0's
                                    )
    return final_df

# load the data
data = load_data(TICKER, N_STEPS, scale=SCALE, split_by_date=SPLIT_BY_DATE, 
                shuffle=SHUFFLE, lookup_step=LOOKUP_STEP, test_size=TEST_SIZE, 
                feature_columns=FEATURE_COLUMNS, ma_periods=MA_PERIODS)

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

# Need to correct scalar for plotting prices without actuals
# get the final dataframe for the testing set
final_df = get_final_df(model, data)
df2 = data['df'].tail(LOOKUP_STEP+1)
df2['forecast'] = 0

indexVals = []
for index in df2.index:
    indexVals.append(index)    
future_prices = predict_gap(model, data, df2, indexVals)
    
future_price = predict(model, data)
print(f"Future $ price after {LOOKUP_STEP} days is {future_price:.2f}")


# plot true/pred prices graph
plot_graph2(final_df, df2)
