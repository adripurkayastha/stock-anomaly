'''
Created on Feb 1, 2015

@author: kooshag
'''

import pandas as pd
import numpy as np



def get_dist(dfin, df_preds, error_type="Euclidean"):
    """
    calculate prediction error
    Arg: original dataframe, predicted dataframe and type of error/distance
    return: error dataframe or "error encountered"
    """
    if (error_type == "Euclidean"):
        df_error = np.sqrt((dfin - df_preds) ** 2)
        # return (df_error.sum(axis=0))
        return df_error
    return "error encountered"


def getChunk(start, win, frq):
    rng = -1
    if frq == 'D':
        rng = pd.date_range(start, periods=win, freq='D')
        # print (rng)
    elif frq == 'W':
        rng = pd.date_range(start, periods=win, freq='W')
    else:
        print('ERROR: the frequency is not D nor W')



def predict_t(df_in, corr):
    """
    Arg:
        a dataframe including time series and a Series
        representing correlation of time series
    return:
        a dataframe including predicted values
    """
    df_pred = df_in.copy()
    for col in df_in.columns:
        ind = 1
        while ind < len(df_in.index):
            actual_val = df_in.iloc[ind, df_in.columns.get_loc(col)]  # the same as (df3[col].iloc[ind])
            pred_val = df_in[col].iloc[ind - 1] * corr[col]
            # print("[{0},{1}: actual= {2} and pred= {3}]".format
            # (ind, df3.columns.get_loc(col), actual_val, pred_val))

            # error1 = np.sqrt((actual_val - pred_val) ** 2) ## probably should be removed

            df_pred.iloc[ind, df_in.columns.get_loc(col)] = pred_val
            ind += 1

    return df_pred


