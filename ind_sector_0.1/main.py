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

# xstart, xend = 1, 20
xstart, xend = 0, 500
df1 = (df.ix[xstart:xend, 1:5])
# df1 = df
outs_ratio = 0.1 # 0.005

""" calculate returns
    NOTE: make sure the last day is the first row in the input dataset is sorted"""

df_in = df1.sort_index()
df3 = df_in.copy()
df3 = (df3 / df3.shift(1)) - 1


df_no_outs = df3.copy()
df_with_outs = df3.copy()

df_outs_ind = df_in.copy()
df_outs_ind[pd.notnull(df_outs_ind)] = 0
df_outs_ind[df_outs_ind.isnull()] = 0

# set outlier index for all prediction methods to 0
pred_outs_inds = df_outs_ind.copy()
df_kdd_ind = df_outs_ind.copy()
df_arima_ind = df_outs_ind.copy()

win_size = 30  # this is set by the input to the algo (very early tests was using winsize 9)

n_chops = int(len(df_no_outs.index) / win_size)

df4 = df3.iloc[:-win_size, :]
lst_sub_df_starts = np.random.choice(df4.index, n_chops)


for sub in lst_sub_df_starts:
    df_sub_start = np.where(df3.index == sub)[0][0]  # we asume the index is unique (i.e. no duplicate datetime)
    df_sub_end = df_sub_start + win_size
    df_sub_in = df_no_outs.iloc[df_sub_start:df_sub_end, :].copy()

    df_sub_out, df_outs_ind = fn.replace_outs2(df_sub_in, df_outs_ind, outs_ratio)
    #print(df_sub_out, df_outs_ind)

    for inx in df_sub_out.index:
        df_with_outs.loc[inx] = df_sub_out.loc[inx]


# store and retrieve the object inout data with artificial outliers
"""import pickle
data1 = df_with_outs, df_outs_ind
output = open('data.pkl', 'wb')
pickle.dump(data1, output)
pkl_file = open('data.pkl', 'rb')
data1 = pickle.load(pkl_file)"""


# pred_outs_inds = df_in.copy() # we set the indexes for all prediction algs in the begining of the program
# pred_outs_inds[pd.notnull(pred_outs_inds)] = 0

strt = 6  # start from the 4th row (i.e. 6-2) row of the input dataframe
while strt < len(df3.index) - win_size:
    strt -= 2
    # call kg prediction for current df
    df_sub = df3.iloc[strt: strt+win_size, :]
    sub_preds_inds = fn.kg_pred(df_sub)[0]

    # update df_inds
    for inx in sub_preds_inds.index:
        pred_outs_inds.loc[inx] = sub_preds_inds.loc[inx]

    print("at indx = {0} date = {1} out of {2}".format(strt, df3.iloc[strt].index, len(df3.index)/win_size))
    strt += win_size


# print(df_with_outs)
# print(df_outs_ind)
prec, rec = fn.get_fmeasure(df_outs_ind, pred_outs_inds)
print(prec, rec)



















