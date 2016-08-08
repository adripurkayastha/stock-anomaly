import pandas as pd

output_path = '/Users/kooshag/Google Drive/code/data/compustat_processed/'
fpath = '/Users/kooshag/Google Drive/code/data/data_SP_daily_nov_2015/'

# fname = 'TEST_dsp500stocks.csv'
fname = 'dsp500stocks.csv'

df = pd.read_csv(fpath + fname, sep=',', header=0, parse_dates=[1], index_col=1,
                 usecols=['PERMNO', 'DATE', 'RET'])

d1 = df['PERMNO'].astype(str).str[0].astype(int)
d2 = df['PERMNO'].astype(str).str[1].astype(int)

df['industry'] = d1.astype(int).multiply(10) + d2.astype(int)
print(df)

industry_codes = {10: "Energy", 15: "Materials"
    , 20: "Industrials", 25: "Consumer Discretionary", 30: "Consumer Staples", 35: "Health Care"
    , 40: "Financials", 45: "Information Technology", 50: "Telecommunication Services", 55: "Utilities"}

selected_inds = {cd: industry_codes[cd] for cd in [10, 25, 30, 40, 45]}

for key in selected_inds:
    df_temp = df[df['industry'] == key]
    df_temp = df_temp.pivot(columns='PERMNO', values='RET')  # pivot to set columns as the companies

    if not df_temp.empty:
        df_temp.to_csv(output_path + selected_inds[key] + ".csv", sep=',', encoding='utf-8')
