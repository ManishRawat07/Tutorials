import yfinance as yf
import sys
TICKER = ""

if len(sys.argv) > 0:
    TICKER = sys.argv[1]

stock = yf.Ticker(TICKER)

# get historical dividends over past 10 years
hist = stock.history(period="10y")

first = True
div_prev = 0
div_risk = False;
for div in stock.dividends:
    if(first):
        first = False
        div_prev = div
    else:
        if(div < div_prev):
            # print (div)
            div_risk = True;
        div_prev = div
print(div_risk)