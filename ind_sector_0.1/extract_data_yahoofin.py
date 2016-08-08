import pandas as pd
from pandas.io.data import DataReader
import pandas.io.data as web
import datetime

output_path = '/Users/kooshag/Google Drive/code/data_market/'

start = datetime.datetime(2016, 6, 21)
end = datetime.datetime(2016, 7, 27)
info_tech_tickers = ["AAPL", "MSFT", "IBM", "ORCL", "GOOGL", "QCOM", "INTC", "FB", "CSCO", "V", "MA",
                         "EBAY", "HPQ", "EMC", "ACN", "TXN", "YHOO", "ADP", "CRM", "ADBE", "CTSH", "MU",
                         "GLW", "AMAT", "TEL", "INTU", "SNDK", "WDC", "MSI", "STX", "APH", "ADI",
                         "FIS", "FISV", "PAYX", "XRX", "ADS", "CA", "SYMC", "JNPR", "NTAP", "XLNX", "ADSK",
                         "CTXS", "KLAC", "LLTC", "NVDA", "AKAM", "CSC", "EA", "LRCX", "MCHP", "RHT",
                         "HRS", "WU", "FFIV", "FSLR", "TDC", "LSI", "TSS", "VRSN", "FLIR", "JBL", "GOOG"]

d = {}
for ticker in info_tech_tickers:
    d[ticker] = DataReader(ticker, "yahoo", start, end)
pan = pd.Panel(d)
df_adj_close = pan.minor_xs('Adj Close') #also use 'Open','High','Low','Adj Close' and 'Volume'


# create a dataframe that has data on only one stock symbol
# df_individual = pan.get('GOOG')

# the dates from the dataframe of just 'GOOG' data
#df_individual.index


df_adj_close.to_csv(output_path+"info_sector_adj_price.csv", sep=',', encoding='utf-8')


