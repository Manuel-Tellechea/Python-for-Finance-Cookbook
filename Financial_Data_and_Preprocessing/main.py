import intrinio_sdk
import matplotlib.pyplot as plt
import warnings
import pandas as pd
import yfinance as yf
import quandl
import intrinio_sdk
import numpy as np
import seaborn as sns
import scipy.stats as scs
import statsmodels.api as sm
import statsmodels.tsa.api as smt

plt.style.use('seaborn')
plt.rcParams['figure.figsize'] = [16, 9]
plt.rcParams['figure.dpi'] = 300
warnings.simplefilter(action='ignore', category=FutureWarning)

# Getting data from different sources
# Download data using Yahoo Finance
# df_yahoo_aapl = yf.download('AAPL', start='2000-01-01', end='2010-12-31', progress=False)


# Download data using Quandl
QUANDL_KEY = 'dC-6qkQyjzDXF8f5u-Mi'
quandl.ApiConfig.api_key = QUANDL_KEY

# df_quandl_aapl = quandl.get(dataset='WIKI/AAPL', start_date='2000-01-01', end_date='2010-12-31')

# Download data using Intrinio
# intrinio_sdk.ApiClient().configuration.api_key['api_key'] = 'OjYyMGQxOTZhM2FjNzhkODBmMGI5MWQxYjVjOTQzNDBj'
# security_api = intrinio_sdk.SecurityApi()

# r = security_api.get_security_stock_prices(identifier='AAPL', start_date='2000-01-01',
#                                            end_date='2010-12-31', frequency='daily', page_size=10000)

# response_list = [x.to_dict() for x in r.stock_prices]
# df_intrinio_aapl = pd.DataFrame(response_list).sort_values('date')
# df_intrinio_aapl.set_index('date', inplace=True)


# Converting prices to returns
# Download the data and keep the adjusted close prices only
data_aapl = yf.download('AAPL', start='2000-01-01', end='2010-12-31', progress=False)
data_aapl = data_aapl.loc[:, ['Adj Close']]
data_aapl.rename(columns={'Adj Close': 'adj_close'}, inplace=True)

# Calculate the simple and log returns using the adjusted close prices
data_aapl['simple_rtn'] = data_aapl.adj_close.pct_change()
data_aapl['log_rtn'] = np.log(data_aapl.adj_close / data_aapl.adj_close.shift(1))

# Accounting for inflation in the returns series
# Create a DataFrame with all possible dates, and left join the prices to it
df_all_dates = pd.DataFrame(index=pd.date_range(start='1999-12-31', end='2010-12-31'))
data = df_all_dates.join(data_aapl[['adj_close']], how='left').fillna(method='ffill').asfreq('M')

# Download the inflation data from Quandl
data_cpi = quandl.get(dataset='RATEINF/CPI_USA', start_date='1999-12-01', end_date='2010-12-31')
data_cpi.rename(columns={'Value': 'cpi'}, inplace=True)

# Merge the inflation data to the prices
data_merged = data.join(data_cpi, how='left')

# Calculate the simple returns and inflation rate
data_merged['simple_rtn'] = data_merged.adj_close.pct_change()
data_merged['inflation_rate'] = data_merged.cpi.pct_change()

# Adjust the returns for inflation
data_merged['real_rtn'] = (data_merged.simple_rtn + 1) / (data_merged.inflation_rate + 1) - 1


# calculate and annualize the monthly realized volatility


def realized_volatility(x):
    return np.sqrt(np.sum(x ** 2))


# Calculate the monthly realized volatility
data_rv = data_aapl.groupby(pd.Grouper(freq='M')).apply(realized_volatility)
data_rv.rename(columns={'log_rtn': 'rv'}, inplace=True)

# Annualize the values
data_rv.rv = data_rv.rv * np.sqrt(12)

# Plot the results
fig, ax = plt.subplots(2, 1, sharex=True)
ax[0].plot(data)
ax[1].plot(data_rv)
# plt.show()


# Plot AAPL's stock prices together with the simple and log returns
fig, ax = plt.subplots(3, 1, figsize=(24, 20), sharex=True)

data_aapl.adj_close.plot(ax=ax[0])
ax[0].set(title='AAPL time series', ylabel='Stock price ($)')
data_aapl.simple_rtn.plot(ax=ax[1])
ax[1].set(ylabel='Simple returns (%)')
data_aapl.log_rtn.plot(ax=ax[2])
ax[2].set(xlabel='Date', ylabel='Log returns (%)')
# plt.show()


# Identifying outliers
df_rolling = data_aapl[['simple_rtn']].rolling(window=21).agg(['mean', 'std'])
df_rolling.columns = df_rolling.columns.droplevel()
df_outliers = data_aapl.join(df_rolling)


def indentify_outliers(row, n_sigmas=3):
    x = row['simple_rtn']
    mu = row['mean']
    sigma = row['std']
    if (x > mu + 3 * sigma) | (x < mu - 3 * sigma):
        return 1
    else:
        return 0


df_outliers['outlier'] = df_outliers.apply(indentify_outliers, axis=1)
outliers = df_outliers.loc[df_outliers['outlier'] == 1, ['simple_rtn']]
fig, ax = plt.subplots()
ax.plot(df_outliers.index, df_outliers.simple_rtn, color='blue', label='Normal')
ax.scatter(outliers.index, outliers.simple_rtn, color='red', label='Anomaly')
ax.set_title("Apple's stock returns")
ax.legend(loc='lower right')
# plt.show()


# Investigating stylized facts of asset returns
# Non-Gaussian distribution of returns
r_range = np.linspace(min(data_aapl.log_rtn), max(data_aapl.log_rtn), num=1000)
mu = data_aapl.log_rtn.mean()
sigma = data_aapl.log_rtn.std()
norm_pdf = scs.norm.pdf(r_range, loc=mu, scale=sigma)

# Plot the histogram
fig, ax = plt.subplots(1, 2, figsize=(16, 8))
# histogram
sns.distplot(data_aapl.log_rtn, kde=False, norm_hist=True, ax=ax[0])
ax[0].set_title('Distribution of AAPL returns', fontsize=16)
ax[0].plot(r_range, norm_pdf, 'g', lw=2, label=f'N({mu:.2f}, {sigma**2:.4f})')
ax[0].legend(loc='upper left')
# Q-Q plot
qq = sm.qqplot(data_aapl.log_rtn.values, line='s', ax=ax[1])
ax[1].set_title('Q-Q plot', fontsize=16)


# Volatility clustering
# Visualize the log returns series
data_aapl.log_rtn.plot(title='Daily AAPL returns')
plt.show()
