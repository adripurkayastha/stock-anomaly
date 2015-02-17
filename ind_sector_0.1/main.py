'''
Created on Nov 23, 2014

@author: kooshag
# 1. extract a df given a window size (return a df)
# 2. calculate centroid and correlations within a given df (return df_corr
# that includes corrs and centroid)
# 3. make predictions given a df_corr (return df_pred)
# 4. calculate error given df, df_pred and error measure (return df_err)
# 5. plotting

'''

import pandas as pd
import numpy as np
import sys


import functions as fn

print(sys.version, pd.__version__, np.__version__)


filePath = '/Users/kooshag/Google Drive/code/data/'
fileName = 'SnP_consumer_dis_daily.csv'

df = pd.read_csv(filePath + fileName,
                 sep=',', header=0, parse_dates=[0], dayfirst=True,
                 index_col=0, skiprows=[0, 1, 2, 4])

xstart, xend = 1, 20
df1 = (df.ix[xstart:xend, 1:5])


""" calculate returns
    NOTE: make sure the last day is the first row in the input dataset is sorted"""

df_in = df1.sort_index()
df3 = df_in.copy()
df3 = (df3 / df3.shift(1)) - 1


df_no_outs = df3.copy()
df_with_outs = df3.copy()



# centroid calculation:
centroid = pd.Series(df3.mean(axis=1), index=df3.index)  # mean of each row

# add centroid to the data frame
df_corr = df3.copy()
df_corr['centroid'] = centroid
df_corr = df_corr.corr()

corr_series = df_corr['centroid']



df_preds = fn.predict_t(df3, corr_series)
print(df_preds)






