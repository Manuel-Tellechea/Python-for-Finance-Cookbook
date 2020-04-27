# Librerías
import pandas as pd
import yfinance as yf
import quandl
import intrinio_sdk
import numpy as np
import cufflinks as cf
import seaborn as sns
import scipy.stats as scs
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import statsmodels.api as sm
import statsmodels.tsa.api as smt
from plotly.offline import iplot

# Importando data con yahoo finance

# df_yahoo = yf.download('AAPL', start='2000-01-01', end='2010-12-31', progress=False)
# print(df_yahoo)

# Importando data con quandl

# QUANDL_KEY = 'gS_NinqJDsEqiAsmxhr3'
# quandl.ApiConfig.api_key = QUANDL_KEY
#
# df_quandl = quandl.get(dataset='WIKI/AAPL', start_date='2000-01-01', end_date='2010-12-31')
# print(df_quandl)

# Importando data con Intrinio
# intrinio_sdk.ApiClient().configuration.api_key['api_key'] = 'OjYyMGQxOTZhM2FjNzhkODBmMGI5MWQxYjVjOTQzNDBj'
# security_api = intrinio_sdk.SecurityApi()
#
# r = security_api.get_security_stock_prices(identifier='AAPL', start_date='2000-01-01',
#                                            end_date='2010-12-31', frequency='daily', page_size=10000)
# response_list = [x.to_dict() for x in r.stock_prices]
# df_intrinio = pd.DataFrame(response_list).sort_values('date')
# df_intrinio.set_index('date', inplace=True)
# print(df_intrinio)

# Converting prices to returns

# df = yf.download('AAPL', start='2000-01-01',
#                  end='2010-12-31', progress=False)
# df = df.loc[:, ['Adj Close']]
# df.rename(columns={'Adj Close': 'adj_close'}, inplace=True)
#
# df['simple_rtn'] = df.adj_close.pct_change()
# df['log_rtn'] = np.log(df.adj_close / df.adj_close.shift(1))

# df_all_dates = pd.DataFrame(index=pd.date_range(start='1999-12-31', end='2010-12-31'))
# df = df_all_dates.join(df[['adj_close']], how='left').fillna(method='ffill').asfreq('M')

# # Inflation rate
# df_cpi = quandl.get(dataset='RATEINF/CPI_USA', start_date='1999-12-01',
#                     end_date='2010-12-31')
# df_cpi.rename(columns={'Value': 'cpi'}, inplace=True)
#
# df_merged = df.join(df_cpi, how='left')
# df_merged['simple_rtn'] = df_merged.adj_close.pct_change()
# df_merged['inflation_rate'] = df_merged.cpi.pct_change()
# df_merged['real_rtn'] = (df_merged.simple_rtn + 1) / (df_merged.inflation_rate + 1) - 1
# #print(df_merged)
#
#
# # Cambiando la frecuencia
#
# def realized_volatility(x):
#     return np.sqrt(np.sum(x ** 2))  # function for calculating the realized volatilit
#
#
# df_rv = df.groupby(pd.Grouper(freq='M')).apply(realized_volatility)
# df_rv.rename(columns={'log_rtn': 'rv'}, inplace=True)  # Calculate the monthly realized volatility
#
# df_rv.rv = df_rv.rv * np.sqrt(12)  # Annualize the values
#
# # fig, ax = plt.subplots(2, 1, sharex=True)
# # ax[0].plot(df)
# # ax[1].plot(df_rv)

# Visualizing time series data

# fig, ax = plt.subplots(3, 1, figsize=(24, 20), sharex=True)
# df.adj_close.plot(ax=ax[0])
# ax[0].set(title='AAPL time series', ylabel='Stock price ($)')
# df.simple_rtn.plot(ax=ax[1])
# ax[1].set(ylabel='Simple returns (%)')
# df.log_rtn.plot(ax=ax[2])
# ax[2].set(xlabel='Date', ylabel='Log returns (%)')
# plt.show()

# df_rolling = df[['simple_rtn']].rolling(window=21).agg(['mean', 'std'])
# df_rolling.columns = df_rolling.columns.droplevel()
# df_outliers = df.join(df_rolling)
#
# def indentify_outliers(row, n_sigmas=3):
#     x = row['simple_rtn']
#     mu = row['mean']
#     sigma = row['std']
#     if (x > mu + 3 * sigma) | (x < mu - 3 * sigma):
#         return 1
#     else:
#         return 0
#
# df_outliers['outlier'] = df_outliers.apply(indentify_outliers, axis=1)
# outliers = df_outliers.loc[df_outliers['outlier'] == 1, ['simple_rtn']]
# fig, ax = plt.subplots()
# ax.plot(df_outliers.index, df_outliers.simple_rtn, color='blue', label='Normal')
# ax.scatter(outliers.index, outliers.simple_rtn, color='red', label='Anomaly')
# ax.set_title("Apple's stock returns")
# ax.legend(loc='lower right')
# plt.show()

df = yf.download(['^GSPC', '^VIX'], start='1985-01-01',
                 end='2018-12-31',progress=False)
df = df[['Adj Close']]
df.columns = df.columns.droplevel(0)
df = df.rename(columns={'^GSPC': 'sp500', '^VIX': 'vix'})

df['log_rtn'] = np.log(df.sp500 / df.sp500.shift(1))
df['vol_rtn'] = np.log(df.vix / df.vix.shift(1))
df.dropna(how='any', axis=0, inplace=True)

# Non-Gaussian distribution of returns

# r_range = np.linspace(min(df.log_rtn), max(df.log_rtn), num=1000)
# mu = df.log_rtn.mean()
# sigma = df.log_rtn.std()
# norm_pdf = scs.norm.pdf(r_range, loc=mu, scale=sigma)
#
# fig, ax = plt.subplots(1, 2, figsize=(16, 8))
#
# # histogram
# sns.distplot(df.log_rtn, kde=False, norm_hist=True, ax=ax[0])
# ax[0].set_title('Distribution of MSFT returns', fontsize=16)
# ax[0].plot(r_range, norm_pdf, 'g', lw=2,
#            label=f'N({mu:.2f}, {sigma**2:.4f})')
# ax[0].legend(loc='upper left');
#
# # Q-Q plot
# qq = sm.qqplot(df.log_rtn.values, line='s', ax=ax[1])
# ax[1].set_title('Q-Q plot', fontsize=16)
# plt.show()
#
# # Volatility clustering
#
# df.log_rtn.plot(title='Daily MSFT returns')
# plt.show()
#
# # Absence of autocorrelation in returns
#
# N_LAGS = 50
# SIGNIFICANCE_LEVEL = 0.05
# acf = smt.graphics.plot_acf(df.log_rtn, lags=N_LAGS,
#                             alpha=SIGNIFICANCE_LEVEL)
# plt.show()
#
# # Small and decreasing autocorrelation in squared/absolute returns
#
# fig, ax = plt.subplots(2, 1, figsize=(12, 10))
# smt.graphics.plot_acf(df.log_rtn ** 2, lags=N_LAGS, alpha=SIGNIFICANCE_LEVEL, ax=ax[0])
# ax[0].set(title='Autocorrelation Plots', ylabel='Squared Returns')
# smt.graphics.plot_acf(np.abs(df.log_rtn), lags=N_LAGS, alpha=SIGNIFICANCE_LEVEL, ax=ax[1])
# ax[1].set(ylabel='Absolute Returns', xlabel='Lag')
# plt.show()

# Leverage effect

df['moving_std_252'] = df[['log_rtn']].rolling(window=252).std()
df['moving_std_21'] = df[['log_rtn']].rolling(window=21).std()

fig, ax = plt.subplots(3, 1, figsize=(18, 15), sharex=True)
df.sp500.plot(ax=ax[0])
ax[0].set(title='S&P500 time series', ylabel='Stock price ($)')
df.log_rtn.plot(ax=ax[1])
ax[1].set(ylabel='Log returns (%)')
df.moving_std_252.plot(ax=ax[2], color='r', label='Moving Volatility 252d')
df.moving_std_21.plot(ax=ax[2], color='g', label='Moving Volatility 21d')
ax[2].set(ylabel='Moving Volatility', xlabel='Date')
ax[2].legend()
plt.show()


corr_coeff = df.log_rtn.corr(df.vol_rtn)
ax = sns.regplot(x='log_rtn', y='vol_rtn', data=df, line_kws={'color': 'red'})
ax.set(title=f'S&P 500 vs. VIX ($\\rho$ = {corr_coeff:.2f})', ylabel='VIX log returns',
       xlabel='S&P 500 log returns')
plt.show()
