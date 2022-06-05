# Tutorials
Tutorials regarding technologies

Knowledge Graphs:
  * Node2Vec contains examples on how to use Node2Vec to analyze knowledge graphs and utilize the gensim word embeddings on the generated model.
  * The StockInfo directory contains an example of how to build a knowledge graph from stock information and then create a node2vec model.
  
Quantum:
  * ShorDecrypt contains an example using IBM of how to utilize quantum computing to decrypt RSA keys
  * Julia contains examples of using Quantum Fourier Transforms and Inverse Quantum Fourier Transforms
  * WaveFunction contains examples of finding the probability distribution without many simulations, and a superposition example that can determine outcomes, a powerful capability.
  * QuantumGraph has an example of modeling a quantum knowledge graph using 5 Qubits in a butterfly structure.
  
TimeSeriesML:
For beginners, this machine learning mastery blog post is a great primer: https://machinelearningmastery.com/how-to-develop-lstm-models-for-time-series-forecasting/
 * StockForecasts has command line options to train, forecast, and evaluate stock forecasts using LSTMs with Tensorflow 2.4 and a GPU.
   * parameters.py contains definitions for the model.  M_STEPS is the number of days back to include in generating a forecast.  This is set at 50 and can be changed.  Each day is a trading day, not a calendar day.  The LOOKUP_STEP is the number of days ahead to forecast.  Currently this is 5, which is 1 week usually.  It's important to shuffle the data and use bi-directional  LSTM cells.  To understand why, change these and compare the forecasts.
    * The loss function can be specified, but we use custom loss function that is slower, but doubly penalizes the model training when the forecasted stock price moves in the wrong direction.
    * It's important to use a sigmoid function in the last layer to avoid generating negative forecasts for wildly changing data.
  * stock_prediction.py contains the code to load data from yahoo finance.  The adjusted closing price is mainly used.  Note that the process should always be to augment raw data with SME data.  To demonstrate this, the open/close/high/low price is augmented with SME metrics of various rolling average periods, bollinger bands, MACD, Momentum, exponential moving average, and simple short term SME price forecasts.
   * The code can be used by typing "python train.py TICKER_SYMBOL days_ahead" to generate a model.  For example, a 20 trading day (4 week) ahead model from CVX can be generate using "python train.py CVX 20".  To obtain the forecast for today use: "python forecast.py CVX 20".  To evaluate the model, use: "python test.py CVX 20".
   * The files ending with MF apply to forecasting mutual funds.  These are different because SME processes are different and forecasts are longer term than for stocks.
   * analyze_MF optimizes a portfolio using an efficient fronteir analysis and a given risk factor to try and obtain returns higher than the lowest volatility point while keeping volatility low.  An example analysis with 3 mutual funds: python .\analyze_MF.py --mfs DSMDX RYVIX WFSJX
  
NLP_ML:
  * Added Facebook Research examples of how to use the fastText library, both for classification and word embeddings.  For word embeddings, the GenSim implementation is used.
