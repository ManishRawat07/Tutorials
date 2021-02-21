import argparse
import numpy as np
import pandas as pd
from yahoo_fin import stock_info as si
import matplotlib.pyplot as plt

#filename to save plot
filename="e:\\temp\\efficientFrontier.png"

# defined command line options
# this also generates --help and error handling
CLI=argparse.ArgumentParser()
CLI.add_argument('--mfs', '--names-list', nargs='*', default=['DSMDX', 'RYVIX', 'WFSJX'])

# parse the command line
args = CLI.parse_args()
# access CLI options
#print("listMFs: %r" % args.mfs)

TICKERS = args.mfs
def load_data(TICKERS):
    out = []
    
    for TICKER in TICKERS:
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
        # load it from yahoo_fin library
        df = si.get_data(TICKER, start_date = "01/01/2016")
        
        # add date as a column
        if "date" not in df.columns:
            df["date"] = df.index
        
        #print(df)
        cols = [(col, TICKER) for col in df.columns]
        df.columns = pd.MultiIndex\
                      .from_tuples(cols,
                                   names=['Attributes', 'Symbols'] )
        out.append(df)
    
    df1 = pd.concat(out, axis=1)
    return df1

# Import data
df = load_data(TICKERS)
df=df.dropna()

# Closing price
df = df['adjclose']

# Log of percentage change
cov_matrix = df.pct_change().apply(lambda x: np.log(1+x)).cov()
#cov_matrix

corr_matrix = df.pct_change().apply(lambda x: np.log(1+x)).corr()
#corr_matrix

# Randomly weighted portfolio's variance
w = []
for ticker in TICKERS:
    w.append(1/len(TICKERS))
port_var = cov_matrix.mul(w, axis=0).mul(w, axis=1).sum().sum()

# Yearly returns for individual companies
ind_er = df.resample('Y').last().pct_change().mean()

# Portfolio returns
port_er = (w*ind_er).sum()

# Volatility is given by the annual standard deviation. We multiply by 252.75 because there are 252.75 trading days/year.
ann_sd = df.pct_change().apply(lambda x: np.log(1+x)).std().apply(lambda x: x*np.sqrt(252.75))

assets = pd.concat([ind_er, ann_sd], axis=1) # Creating a table for visualising returns and volatility of assets
assets.columns = ['Returns', 'Volatility']

p_ret = [] # Define an empty array for portfolio returns
p_vol = [] # Define an empty array for portfolio volatility
p_weights = [] # Define an empty array for asset weights

num_assets = len(df.columns)
num_portfolios = 10000

for portfolio in range(num_portfolios):
    weights = np.random.random(num_assets)
    weights = weights/np.sum(weights)
    p_weights.append(weights)
    returns = np.dot(weights, ind_er) # Returns are the product of individual expected returns of asset and its 
                                      # weights 
    p_ret.append(returns)
    var = cov_matrix.mul(weights, axis=0).mul(weights, axis=1).sum().sum()# Portfolio Variance
    sd = np.sqrt(var) # Daily standard deviation
    ann_sd = sd*np.sqrt(250) # Annual standard deviation = volatility
    p_vol.append(ann_sd)
    
data = {'Returns':p_ret, 'Volatility':p_vol}

for counter, symbol in enumerate(df.columns.tolist()):
    #print(counter, symbol)
    data[symbol+' weight'] = [w[counter] for w in p_weights]
    
portfolios  = pd.DataFrame(data)

# Plot efficient frontier
#portfolios.plot.scatter(x='Volatility', y='Returns', marker='o', s=10, alpha=0.3, grid=True, figsize=[10,10])

min_vol_port = portfolios.iloc[portfolios['Volatility'].idxmin()]
# idxmin() gives us the minimum value in the column specified.   

# plotting the minimum volatility portfolio
#plt.subplots(figsize=[10,10])
#plt.scatter(portfolios['Volatility'], portfolios['Returns'],marker='o', s=10, alpha=0.3)
#plt.scatter(min_vol_port[1], min_vol_port[0], color='r', marker='*', s=500)

# Finding the optimal portfolio
rf = 0.157 # risk factor
optimal_risky_port = portfolios.iloc[((portfolios['Returns']-rf)/portfolios['Volatility']).idxmax()]


allTickers = ""
for TICKER in TICKERS:
    allTickers += TICKER + " "

# Plotting optimal portfolio
plt.figure(figsize=(18,18))
plt.title("Efficient Fronteir Mutual Fund Analysis ("+allTickers.rstrip()+")", fontsize=16)
    
plt.scatter(portfolios['Volatility'], portfolios['Returns'],marker='o', s=10, alpha=0.3)
plt.scatter(min_vol_port[1], min_vol_port[0], color='r', marker='*', s=500)
plt.scatter(optimal_risky_port[1], optimal_risky_port[0], color='g', marker='*', s=500)
plt.xlabel("Volatility", fontsize=16)
plt.ylabel("Returns", fontsize=16)
plt.savefig(filename)

print(optimal_risky_port)



