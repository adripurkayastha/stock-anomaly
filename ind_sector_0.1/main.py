__author__ = 'kooshag'

'''
Created on Nov 23, 2014

1. extract a df given a window size (return a df)
2. calculate centroid and correlations within a given df (return df_corr
   that includes corrs and centroid)
3. make predictions given a df_corr (return df_pred)
4. calculate error given df, df_pred and error measure (return df_err)
5. plotting
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

df_outs_ind = df_in.copy()
df_outs_ind[pd.notnull(df_outs_ind)] = 0

win_size = 9  # this is set by the input to the algo
n_chops = int(len(df_no_outs.index) / win_size)

df4 = df3.iloc[:-win_size, :]
lst_sub_df_starts = np.random.choice(df4.index, n_chops)

for sub in lst_sub_df_starts:
    df_sub_start = np.where(df3.index == sub)[0][0]  # we asume the index is unique (i.e. no duplicate datetime)
    df_sub_end = df_sub_start + win_size
    df_sub_in = df_no_outs.iloc[df_sub_start:df_sub_end, :].copy()

    df_sub_out, df_outs_ind = fn.replace_outs2(df_sub_in, df_outs_ind, 0.1)
    #print(df_sub_out, df_outs_ind)

    for inx in df_sub_out.index:
        df_with_outs.loc[inx] = df_sub_out.loc[inx]



# centroid calculation:
centroid = pd.Series(df3.mean(axis=1), index=df3.index)  # mean of each row

# add centroid to the data frame
df_corr = df3.copy()
df_corr['centroid'] = centroid
df_corr = df_corr.corr()

corr_series = df_corr['centroid']

df_error = df3.copy()
df_preds = fn.predict_t(df3, corr_series)
pred_outs = fn.get_pred_outs(df_preds, df3)


print(df_with_outs)
print(df_outs_ind)








