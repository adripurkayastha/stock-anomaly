'''
Created on Feb 1, 2015

@author: kooshag
'''

import pandas as pd
import numpy as np
import scipy.stats as stats
import random as rnd

from sklearn.metrics import classification_report, recall_score, precision_score
from sklearn.neighbors import NearestNeighbors
import statsmodels.tsa.api as tsa



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


def replace_outs(df, numOuts, df_outs_ind):
    """
    This has been replaced with "replace_outs2"
    """
    df_out = df.copy()
    out_row_inds, out_col_inds = np.random.randint(0, len(df_outs_ind.index), numOuts), \
                                 np.random.randint(0, len(df_outs_ind.columns), numOuts)

    for row, col in zip(out_row_inds, out_col_inds):
        array_col = df.iloc[:, col].dropna()
        z_score, p_val = stats.normaltest(array_col)

        if p_val > 0.05:  # this means the distribution is normal
            eps = 0.002 * np.random.random_sample(1) - 0.001  # epsilon is a random float in [-0.001, 0.001]
                                                                # *** this threshold should be set in experiments
            df_out.iloc[row, col] = 3 * df.iloc[:, col].std() + eps
            # print("for row {0} and column {1} we have {2} and real val is {3}".format(row, col, df_out.iloc[row, col], df_in.iloc[row, col]))
            df_outs_ind.iloc[row, col] = 1

        else:
            q1, q3, iqr = tukey_vals(array_col)
            tukeyHL = [array_col.mean() + q3 + (3 * iqr), array_col.mean() - q1 - (3 * iqr)]
            df_out.iloc[row, col] = rnd.sample(tukeyHL, 1)
            df_outs_ind.iloc[row, col] = 1

    return df_out, df_outs_ind


def tukey_vals(lst_in):
    """
    Arg:
        input array of numbers
    Return:
        Tukey's Q1, Q3 and IQR
    """
    try:
        np.sum(lst_in)
    except TypeError:
        print('Error: you must provide a list or array of only numbers')
    q1 = stats.scoreatpercentile(lst_in[~np.isnan(lst_in)], 25)
    q3 = stats.scoreatpercentile(lst_in[~np.isnan(lst_in)], 75)
    iqr = q3 - q1
    return q1, q3, iqr


def replace_outs2(df, df_outs_ind, outs_ratio):
    """
    Args:
        df: dataframe representing the input dataset
        df_outs_ind: a matrix of 0/1 representing index of injected outliers before calling this function
        outs_ratio: the ratio of data points that should be replaced with outliers in each window
    Return:
        df_out: a dataframe that includes injected outliers in the input data
        df_outs_ind: a dataframe of 0/1 representing index of injected outliers after calling this function

    """
    df_out = df.copy()
    n_outs = int(outs_ratio * df.count().sum())
    out_indx, out_cols = np.random.choice(df.index, n_outs), np.random.choice(df.columns, n_outs)

    for row, col in zip(out_indx, out_cols):
        array_col = df.loc[:, col].dropna()

        if len(array_col) < 9:
            continue
        z_score, p_val = stats.normaltest(array_col)

        if p_val > 0.05:  # this means the distribution is normal
            eps = 0.002 * np.random.random_sample(1) - 0.001  # epsilon is a random float in [-0.001, 0.001]
                                                                # *** this threshold should be set in experiments
            # eps += 1000 # this was for testing
            df_out.loc[row, col] = 3 * df.loc[:, col].std() + eps
            # print("for row {0} and column {1} we have {2} and real val is {3}".format(row, col, df_out.iloc[row, col], df_in.iloc[row, col]))
            df_outs_ind.loc[row, col] = 1

        else:
            q1, q3, iqr = tukey_vals(array_col)
            tukeyHL = [array_col.mean() + q3 + (3 * iqr), array_col.mean() - q1 - (3 * iqr)]
            df_out.loc[row, col] = rnd.sample(tukeyHL, 1)[0]
            df_outs_ind.loc[row, col] = 1

    return df_out, df_outs_ind


def get_pred_outs(preds, df_in):
    """
    Args:
        preds: dataframe representing prediction values
        df_in: dataframe representing input data
    Return:
        df_TF: a dataframe of 0/1 representing index of data points that are predicted as outliers
    """
    df_dist = get_dist(df_in, preds)

    df_TF = df_in.copy()
    df_TF.fillna(0, inplace=True)
    df_TF[np.isfinite(df_TF)] = 0

    for col in df_dist.columns:
        for indx in df_dist.index:
            if df_dist[col].loc[indx] > df_in[col].std():
                df_TF[col].loc[indx] = 1
                #print(col, indx, df_TF[col].loc[indx])
            #print(indx, col, df_dist.loc[indx, col], df_in[col].std())

    return df_TF


def get_fmeasure(df, df_preds):
    y = df_to_arr(df)# these are df.values
    y_hat = df_to_arr(df_preds)# this would be the predictions

    target_names = [0, 1]
    report = classification_report(y, y_hat, target_names)
    print(report)

    rec = recall_score(y, y_hat)
    prec = precision_score(y, y_hat)
    return prec, rec


def df_to_arr(df):
    arr = []
    for col in df.columns:
        for ix in df.index:
            arr.append(df[col].loc[ix])

    return arr


def kg_pred(df_sub):
    df = df_sub
    # centroid calculation:
    centroid = pd.Series(df.mean(axis=1), index=df.index)  # mean of each row

    # add centroid to the data frame
    df_corr = df.copy()
    df_corr['centroid'] = centroid
    df_corr = df_corr.corr()
    corr_series = df_corr['centroid']

    df_error = df.copy() # should be removed
    sub_preds = predict_t(df, corr_series)
    sub_pred_outs_inds = get_pred_outs(sub_preds, df)
    return sub_pred_outs_inds, sub_preds


def knn_preds(df, df_knn_inds, k=4):
    for idx in df.index:
        # X = df.loc[idx].values
        row = df.loc[idx]

        row2 = row.dropna()

        X = row2.values

        if len(X) < k:
            continue

        X_tmp = np.reshape(X, (len(X), 1))  # convert X to a 2D array
        X_2D = np.zeros((len(X), 2))
        X_2D[:, 1:] = X_tmp  #  X_2D includes values of tickers at one time stamp (e.g. values of WALT DISNEY  COMCAST 'A'  HOME DEPOT  at 2014-04-14 )

        nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(X_2D)
        distances, indices = nbrs.kneighbors(X_2D)
        # print(indices)
        # print(distances)

        # dists = scp.spatial.distance.cdist(X_2D, X_2D, 'euclidean')
        sum_arr = distances.sum(axis=1) #  sum over rows gives sum kNN for each ticker (e.g. "WALT DISNEY")
                                    # at timestamp idx
        mu = sum_arr.mean()
        sigma = sum_arr.std()
        for sum, row_ind in zip(sum_arr, row2.index):
            if sum > 1 * sigma + mu: # could be changed to 2*sigma + mu
                df_knn_inds[row_ind].loc[idx] = 1

    return df_knn_inds


def arima_pred(df, arima_inds):

    arima_preds = df.copy()
    for col in df.columns:

        arr = df[col].dropna().values

        arma = tsa.ARMA(arr, order=(0, 1, 0))
        results = arma.fit()
        res = results.predict(0, len(arr)-1)

        for item, ind in zip(res, arima_preds.index):
            if np.isnan(arima_preds[col].loc[ind]):
                continue
            arima_preds[col].loc[ind] = item

        ts = arima_preds[col]
        mu = ts.mean()
        sigma = ts.std()
        for idx in ts.index:
            if ts.loc[idx] > 3 * sigma + mu:    # could be changed to 2*sigma + mu
                arima_inds[col].loc[idx] = 1

    return arima_inds


def arima_pred2(df, arima_inds):


    # arima_preds = df.copy()
    for col in df.columns:
        sigma = df[col].std()
        mu = df[col].mean()

        pred = df[col].copy()
        for ind in df[col].index:
            pred.loc[ind] = df[col].loc[ind] + np.random.normal(mu, sigma)

        arima_preds = pred.shift(1)

        for ind in arima_preds.index:
            if arima_preds[ind] > 3 * sigma + mu:    # could be changed to 2*sigma + mu
                arima_inds[col].loc[ind] = 1

    return arima_inds


def f2(p,r):
    return 5 * p * r / ((4*p)+r)
