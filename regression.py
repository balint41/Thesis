#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 18:07:21 2023

@author: horvathbalint
"""
import pandas as pd
import numpy as np
import math
import yfinance as yf
import seaborn as sns
import matplotlib.pyplot as plt
from fastparquet import write
import os
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.graphics.tsaplots import plot_acf
from arch.unitroot import PhillipsPerron
import statsmodels.formula.api as smf
from statsmodels.tsa.stattools import adfuller
from stargazer.stargazer import Stargazer
import scipy.stats as stats
import statsmodels.stats.api as sms
from statsmodels.compat import lzip

#%% 3 different ways to compute realised / historical volatility.

#Standard deviation (Close-to-Close)
def standard_deviation(price_data, window=30, trading_periods=252, clean=True):

    log_return = (price_data["Close"] / price_data["Close"].shift(1)).apply(np.log)

    result = log_return.rolling(window=window, center=False).std() * math.sqrt(
        trading_periods
    )

    if clean:
        return result.dropna()
    else:
        return result

#Parkinson volatility
def parkinson(price_data, window=30, trading_periods=252, clean=True):
    rs = (1.0 / (4.0 * math.log(2.0))) * (
        (price_data["High"] / price_data["Low"]).apply(np.log)
    ) ** 2.0
    
    def f(v):
        return(trading_periods*v.mean())**0.5
    
    result=rs.rolling(window=window, center=False).apply(func=f)
    
    if clean:
        return result.dropna()
    else:
        return result 
    
#Garman-Klass volatility 
def garman_klass(price_data, window=30, trading_periods=252, clean=True):

    log_hl = (price_data["High"] / price_data["Low"]).apply(np.log)
    log_co = (price_data["Close"] / price_data["Open"]).apply(np.log)

    rs = 0.5 * log_hl ** 2 - (2 * math.log(2) - 1) * log_co ** 2

    def f(v):
        return (trading_periods * v.mean()) ** 0.5

    result = rs.rolling(window=window, center=False).apply(func=f)

    if clean:
        return result.dropna()
    else:
        return result
#%% Underlying market data (Index prices).  
"""Data source: Bloomberg (volume, close price), YFinance (high, low, open)"""   

BEL20=pd.read_excel('/Users/horvathbalint/Documents/Pénzügy_MSc/Szakdolgozat/Adatok/Index/BEL20.xlsx')
BEL20['Date'] = pd.to_datetime(BEL20['Date'])
BEL20.set_index('Date', inplace=True)
BEL20['log_ret']=np.log(BEL20['Close']/BEL20['Close'].shift(1))
BEL20['mavg_volume']=BEL20['Volume'].rolling(window=21).mean()
BEL20['sd_vola_252']=BEL20['log_ret'].rolling(window=252).std() * np.sqrt(252)
#Bloomberg data does not contain open, low, high data - so YFinance needed.
#However volume data of Bloomberg is more accurate, so we merge these two.
data=yf.download("^BFX", start="2001-01-01", end="2021-01-01")
BEL20['sd_vola']=standard_deviation(data)
BEL20['parkinson_vola']=parkinson(data)
BEL20['gk_vola']=garman_klass(data)
# Plot them 
#fig, ax = plt.subplots(figsize=(12, 8))
#sns.lineplot(data=BEL20[['sd_vola', 'parkinson_vola', 'gk_vola']], ax=ax, palette=['#3297a8', '#c2185b', '#7b9095'])

####

#DAX partly prepared in excel 
DAX=pd.read_excel('/Users/horvathbalint/Documents/Pénzügy_MSc/Szakdolgozat/Adatok/Index/DAX.xlsx')
DAX.set_index('Date', inplace=True)
data=yf.download("^GDAXI", start="2001-01-01", end="2021-01-01")
DAX['sd_vola']=standard_deviation(data)
DAX['parkinson_vola']=parkinson(data)
DAX['gk_vola']=garman_klass(data)

###

HEX=pd.read_excel('/Users/horvathbalint/Documents/Pénzügy_MSc/Szakdolgozat/Adatok/Index/HEX.xlsx')
HEX['log_ret']=np.log(HEX['Close']/HEX['Close'].shift(1))
HEX['Date'] = pd.to_datetime(HEX['Date'])
HEX.set_index('Date', inplace=True)
HEX['sd_vola_252']=HEX['log_ret'].rolling(window=252).std() * np.sqrt(252)
HEX['mavg_volume']=HEX['Volume'].rolling(window=21).mean()
data=yf.download("^HEX", start="2001-01-01", end="2021-01-01")
HEX['sd_vola']=standard_deviation(data)
HEX['parkinson_vola']=parkinson(data)
HEX['gk_vola']=garman_klass(data)

###

SMI=pd.read_excel('/Users/horvathbalint/Documents/Pénzügy_MSc/Szakdolgozat/Adatok/Index/SMI.xlsx')
SMI['log_ret']=np.log(SMI['Close']/SMI['Close'].shift(1))
SMI['Date'] = pd.to_datetime(SMI['Date'])
SMI.set_index('Date', inplace=True)
SMI['sd_vola_252']=SMI['log_ret'].rolling(window=252).std() * np.sqrt(252)
SMI['mavg_volume']=SMI['Volume'].rolling(window=21).mean()
data=yf.download("^SSMI", start="2001-01-01", end="2021-01-01")
SMI['sd_vola']=standard_deviation(data)
SMI['parkinson_vola']=parkinson(data)
SMI['gk_vola']=garman_klass(data)

###

UKX=pd.read_excel('/Users/horvathbalint/Documents/Pénzügy_MSc/Szakdolgozat/Adatok/Index/UKX.xlsx')
UKX['log_ret']=np.log(UKX['Close']/UKX['Close'].shift(1))
UKX['Date'] = pd.to_datetime(UKX['Date'])
UKX.set_index('Date', inplace=True)
UKX['mavg_volume']=UKX['Volume'].rolling(window=21).mean()  
data=yf.download("^FTSE", start="2001-01-01", end="2021-01-01")
UKX['sd_vola']=standard_deviation(data)
UKX['parkinson_vola']=parkinson(data)
UKX['gk_vola']=garman_klass(data)

del data

#%% Expand with control variables.
c=pd.read_excel('/Users/horvathbalint/Documents/Pénzügy_MSc/Szakdolgozat/Adatok/Control/ESI_ECB.xlsx')
#contains the European Sentiment Indicator, the BoE and the ECB rating decision history (dummies)
vix=yf.download("^VIX", start="2002-01-01", end="2019-12-31") #proxy of the global volatility in the model. 
vix=vix.reset_index()
vix=vix[['Date','Adj Close']]
vix=vix.rename(columns={'Adj Close':'VIX'})
# fg=pd.read_csv('/Users/horvathbalint/Documents/Pénzügy_MSc/Szakdolgozat/Adatok/Control/fear-greed.csv')
# fg=fg[['Date','Fear Greed']]
# fg['Date'] = pd.to_datetime(fg['Date'])

DAX=pd.merge(DAX, vix, on="Date", how="left")
DAX=pd.merge(DAX, c, on="Date", how="left")
# DAX=pd.merge(DAX, fg, on="Date", how="left")
DAX['VIX'] = DAX['VIX'].fillna(DAX['VIX'].rolling(window=21, min_periods=1).mean())

SMI=pd.merge(SMI, vix, on="Date", how="left")
SMI=pd.merge(SMI, c, on="Date", how="left")
# SMI=pd.merge(SMI, fg, on="Date", how="left")
SMI['VIX'] = SMI['VIX'].fillna(SMI['VIX'].rolling(window=21, min_periods=1).mean())

UKX=pd.merge(UKX, vix, on="Date", how="left")
UKX=pd.merge(UKX, c, on="Date", how="left")
# UKX=pd.merge(UKX, fg, on="Date", how="left")
UKX['VIX'] = UKX['VIX'].fillna(UKX['VIX'].rolling(window=21, min_periods=1).mean())

BEL20=pd.merge(BEL20, vix, on="Date", how="left")
BEL20=pd.merge(BEL20, c, on="Date", how="left")
# BEL20=pd.merge(BEL20, fg, on="Date", how="left")
BEL20['VIX'] = BEL20['VIX'].fillna(BEL20['VIX'].rolling(window=21, min_periods=1).mean())

HEX=pd.merge(HEX, vix, on="Date", how="left")
HEX=pd.merge(HEX, c, on="Date", how="left")
# HEX=pd.merge(HEX, fg, on="Date", how="left")
HEX['VIX'] = HEX['VIX'].fillna(HEX['VIX'].rolling(window=21, min_periods=1).mean())

del c, vix
# del fg

#%% Option market data.
# directory = "/Users/horvathbalint/Data/Options/Raw_PQ"
# file_paths = []

# for filename in os.listdir(directory):
#     if filename.endswith(".parquet"):
#         path = os.path.join(directory, filename)
#         file_paths.append(path)

# dfs = []
# for path in file_paths:
#     df = pd.read_parquet(path)
#     dfs.append(df)

# df = pd.concat(dfs)

# opt=pd.read_parquet('/Users/horvathbalint/Data/Options/Medium/OptionID.parq')

# df['Date']=pd.to_timedelta(df['Date'], unit='D')+pd.Timestamp('1960-1-1')
# df=df.drop(['Currency' , 'Bid' , 'BidTime' , 'UnderlyingBid' , 'Ask' , 
#             'AskTime' , 'UnderlyingAsk' , 'Vega' , 'Theta' , 'Last' , 
#             'LastTime' , 'ReferenceExchange', 'CalculationPrice'], axis=1, inplace=False)

# """Implied volatility and the greeks was set to -99.99 if IV calculation fails to 
# converge or if the option is a "special settlement" or the midpoint of the
# bid/ask price is below instrinic value. """

# df = df[~np.isclose(df['ImpliedVolatility'], -99.989998, rtol=1e-5)]

# """We are looking for these 7 European stock indices: """
# def get_symbol(SecurityID):
#     if SecurityID == 501271.0:
#         return 'AEX'
#     elif SecurityID==506496.0:
#         return 'DAX'
#     elif SecurityID==508037.0:
#         return 'CAC'
#     elif SecurityID==506522.0 or SecurityID==707745.0:
#         return 'SMI'
#     elif SecurityID==510399.0:
#         return 'HEX'
#     elif SecurityID==506528.0:
#         return 'UKX'
#     elif SecurityID==506552.0:
#         return 'BEL20'
#     else:
#         return 'NA'
    
# df = df.reset_index()
# df['Ticker'] = df['SecurityID'].apply(get_symbol)

# """Input contract size from the OptionHistory file"""
# merged_df = pd.merge(df, opt, on='OptionID', how='left')
# df=merged_df.drop(['index','SecurityID_y', 'Strike', 'CallPut', 'OptionStyle', 'ExerciseStyle', 'Expiration'], axis=1, inplace=False)

# """Calculating Gamma exposure based on SqueezeMetrics Research and Sergei Perfiliev"""
# get_call_put = lambda x: 1 if x['Delta'] > 0 else -1
# df['CallPut'] = df.apply(get_call_put, axis=1)

# df['GEX']=df['Gamma']*df['OpenInterest']*df['ContractSize']*df['UnderlyingLast']*df['CallPut']
# df=df.groupby(['Date', 'Ticker'])[['GEX', 'Volume']].sum() 

# write('/Users/horvathbalint/Data/Options/Medium/optionprice_final.parq', df)

# del dfs, directory, file_paths, filename, merged_df, opt, path

#%% Merging the underlying market and the option market data. 

"""Read in the option market data"""
#optionally speed up the code by reading in the option market data from file
df=pd.read_parquet('/Users/horvathbalint/Data/Options/Medium//optionprice_final.parq')
df=df.rename(columns={'Volume':'OptVolume'})

###

df_dax = df.loc[(slice(None), 'DAX'), :]
DAX=pd.merge(DAX, df_dax, on="Date", how="left")
DAX['O/S']=DAX['OptVolume']/DAX['Volume']
DAX['gamma_proxy']=DAX['GEX']/DAX['mavg_volume']
DAX = DAX.loc[(DAX["Date"] >= "2002-01-01") & (DAX["Date"] <= "2019-12-31")]
mavg=DAX['gamma_proxy'].rolling(window=7, min_periods=1).mean()
DAX['gamma_proxy' ]=DAX['gamma_proxy'].fillna(mavg)
DAX[['sd_vola_252','sd_vola', 'parkinson_vola', 'gk_vola']].fillna(method='ffill')
DAX=DAX.dropna()

###

df_smi=df.loc[(slice(None), 'SMI'), :]
SMI=pd.merge(SMI, df_smi, on="Date", how="left")
SMI['O/S']=SMI['OptVolume']/SMI['Volume']
SMI['gamma_proxy']=SMI['GEX']/SMI['mavg_volume']
SMI = SMI.loc[(SMI["Date"] >= "2002-01-01") & (SMI["Date"] <= "2019-12-31")]
mavg=SMI['gamma_proxy'].rolling(window=7, min_periods=1).mean()
SMI['gamma_proxy' ]=SMI['gamma_proxy'].fillna(mavg)
SMI[['sd_vola_252','sd_vola', 'parkinson_vola', 'gk_vola']].fillna(method='ffill')
SMI=SMI.dropna()


###

df_bel=df.loc[(slice(None), 'BEL20'), :]
BEL20=pd.merge(BEL20, df_bel, on="Date", how="left")
BEL20['O/S']=BEL20['OptVolume']/BEL20['Volume']
BEL20['gamma_proxy']=BEL20['GEX']/BEL20['mavg_volume']
BEL20 = BEL20.loc[(BEL20["Date"] >= "2002-01-01") & (BEL20["Date"] <= "2019-12-31")]
mavg=BEL20['gamma_proxy'].rolling(window=7, min_periods=1).mean()
BEL20['gamma_proxy' ]=BEL20['gamma_proxy'].fillna(mavg)
BEL20[['sd_vola_252','sd_vola', 'parkinson_vola', 'gk_vola']].fillna(method='ffill')
BEL20=BEL20.dropna()

###

df_hex=df.loc[(slice(None), 'HEX'), :]
HEX=pd.merge(HEX, df_hex, on="Date", how="left")
HEX['O/S']=HEX['OptVolume']/HEX['Volume']
HEX['gamma_proxy']=HEX['GEX']/HEX['mavg_volume']
HEX = HEX.loc[(HEX["Date"] >= "2002-01-01") & (HEX["Date"] <= "2019-12-31")]
mavg=HEX['gamma_proxy'].rolling(window=7, min_periods=1).mean()
HEX['gamma_proxy' ]=HEX['gamma_proxy'].fillna(mavg)
HEX[['sd_vola', 'parkinson_vola', 'gk_vola']].fillna(method='ffill')
HEX=HEX.dropna()

###

df_ukx=df.loc[(slice(None), 'UKX'), :]
UKX=pd.merge(UKX, df_ukx, on="Date", how="left")
UKX['O/S']=UKX['OptVolume']/UKX['Volume']
UKX['gamma_proxy']=UKX['GEX']/UKX['mavg_volume']
UKX = UKX.loc[(UKX["Date"] >= "2002-01-01") & (UKX["Date"] <= "2019-12-31")]
mavg=UKX['gamma_proxy'].rolling(window=7, min_periods=1).mean()
UKX['gamma_proxy' ]=UKX['gamma_proxy'].fillna(mavg)
UKX[[ 'sd_vola', 'parkinson_vola', 'gk_vola']].fillna(method='ffill')
UKX=UKX.dropna()

del df_bel, df_dax, df_smi, df_ukx, df_hex, mavg, df

#%% Descriptive statistics and plots. 

DAX[['gamma_proxy', 'log_ret', 'gk_vola']].describe().T
HEX[['gamma_proxy', 'log_ret', 'gk_vola']].describe().T
SMI[['gamma_proxy', 'log_ret', 'gk_vola']].describe().T
UKX[['gamma_proxy', 'log_ret',  'gk_vola']].describe().T
pd.set_option('display.max_columns', None)
BEL20[['gamma_proxy', 'log_ret', 'gk_vola']].describe(include='all').T

###

sns.set(style='whitegrid', palette=['#3297a8', '#c2185b'])
fig, ax = plt.subplots(figsize=(12, 8))
sns.lineplot(x='Date', y='gamma_proxy', data=DAX, label='gamma proxy', ax=ax)
sns.lineplot(x='Date', y='gk_vola', data=DAX, label='Garman-Klass volatilitás', ax=ax)
plt.title('DAX: Nettó gamma kitettség és volatilitás', size=16)
plt.legend()
plt.gca().set(xlabel='', ylabel='')
plt.show()

###

fig, ax = plt.subplots(figsize=(12, 8))
sns.lineplot(x='Date', y='gamma_proxy', data=SMI, label='gamma proxy', ax=ax)
sns.lineplot(x='Date', y='gk_vola', data=SMI, label='Garman-Klass volatilitás', ax=ax)
plt.title('SMI: Nettó gamma kitettség és volatilitás', size=16)
plt.legend()
plt.gca().set(xlabel='', ylabel='')
plt.show()

###

sns.set(style='white', palette=['#3297a8', '#c2185b'])
fig, ax = plt.subplots(figsize=(12, 8))
ax2 = ax.twinx()

sns.lineplot(x='Date', y='gamma_proxy', data=UKX, label='gamma proxy', ax=ax, color='#3297a8', linewidth=2)
sns.lineplot(x='Date', y='gk_vola', data=UKX, label='Garman-Klass volatilitás', ax=ax2, color='#c2185b', linewidth=2)

plt.title('FTSE-100: Nettó gamma kitettség és volatilitás', size=16)
ax.legend(loc='lower right')
ax.set_xlabel('')
ax.set_ylabel('')
ax2.set_ylabel('')
plt.show()

###

sns.set(style='whitegrid', palette=['#3297a8', '#c2185b'])
fig, ax = plt.subplots(figsize=(12, 8))
sns.lineplot(x='Date', y='gamma_proxy', data=BEL20, label='gamma proxy', ax=ax)
sns.lineplot(x='Date', y='gk_vola', data=BEL20, label='Garman-Klass volatilitás', ax=ax)
plt.title('BEL20: Nettó gamma kitettség és volatilitás', size=16)
plt.legend()
plt.gca().set(xlabel='', ylabel='')
plt.show()

###

fig, ax = plt.subplots(figsize=(12, 8))
sns.lineplot(x='Date', y='gamma_proxy', data=HEX, label='gamma proxy', ax=ax)
sns.lineplot(x='Date', y='gk_vola', data=HEX, label='Garman-Klass volatilitás', ax=ax)
plt.title('HEX: Nettó gamma kitettség és volatilitás', size=16)
plt.legend()
plt.gca().set(xlabel='', ylabel='')
plt.show()

del fig, ax, ax2

#%%ACF, PACF
"""Serial autocorrelation of the studied variables."""

plot_acf(DAX['gamma_proxy'], lags=70)
plt.title('DAX: gamma proxy ACF')
plt.show()
plot_acf(DAX['gk_vola'], lags=70)
plt.title('DAX: volatilitás ACF')
plt.show()
plot_pacf(DAX['gamma_proxy'], lags=70)
plt.title('DAX: gamma proxy PACF')
plt.show()
plot_pacf(DAX['gk_vola'], lags=70)
plt.title('DAX: volatilitás PACF')
plt.show()

###

plot_acf(SMI['gamma_proxy'], lags=70)
plt.title('SMI: gamma proxy ACF')
plt.show()

###

plot_acf(BEL20['gamma_proxy'], lags=70)
plt.title('BEL20: gamma proxy ACF')
plt.show()

###

plot_acf(HEX['gamma_proxy'], lags=70)
plt.title('HEX: gamma proxy ACF')
plt.show()

###

plot_acf(UKX['gamma_proxy'], lags=70)
plt.title('UKX: gamma proxy ACF')
plt.show()

#There is significant autocorrelation in the time series. 

#%% Unit root test. 
#DAX: volatility is on the border of rejecting the H0, under the given lags, with no trend. 
print(PhillipsPerron(DAX["gamma_proxy"], lags=32, test_type="rho", trend="n"))
print(PhillipsPerron(DAX["gamma_proxy"], lags=32, test_type="rho", trend="c"))
print(PhillipsPerron(DAX["gamma_proxy"], lags=32, test_type="rho", trend="ct"))

print(PhillipsPerron(DAX["gk_vola"], lags=32, test_type="rho", trend="n")) #hits the 5% critical value!
print(PhillipsPerron(DAX["gk_vola"], lags=32, test_type="rho", trend="c"))
print(PhillipsPerron(DAX["gk_vola"], lags=32, test_type="rho", trend="ct"))

print(PhillipsPerron(DAX["log_ret"], lags=32, test_type="rho", trend="n"))
print(PhillipsPerron(DAX["log_ret"], lags=32, test_type="rho", trend="c"))
print(PhillipsPerron(DAX["log_ret"], lags=32, test_type="rho", trend="ct"))

print(adfuller(DAX['gk_vola'])) #2,5% p-value

###
#UKX: stationary. 
print(PhillipsPerron(UKX["gamma_proxy"], lags=32, test_type="rho", trend="n"))
print(PhillipsPerron(UKX["gamma_proxy"], lags=32, test_type="rho", trend="c"))
print(PhillipsPerron(UKX["gamma_proxy"], lags=32, test_type="rho", trend="ct"))

print(PhillipsPerron(UKX["gk_vola"], lags=32, test_type="rho", trend="n"))
print(PhillipsPerron(UKX["gk_vola"], lags=32, test_type="rho", trend="c"))
print(PhillipsPerron(UKX["gk_vola"], lags=32, test_type="rho", trend="ct"))

print(PhillipsPerron(UKX["log_ret"], lags=32, test_type="rho", trend="n"))
print(PhillipsPerron(UKX["log_ret"], lags=32, test_type="rho", trend="c"))
print(PhillipsPerron(UKX["log_ret"], lags=32, test_type="rho", trend="ct"))

print(adfuller(UKX['parkinson_vola']))
###
#HEX: volatility is on the border of rejecting the H0, under the given lags, with no trend.
print(PhillipsPerron(HEX["gamma_proxy"], lags=32, test_type="rho", trend="n"))
print(PhillipsPerron(HEX["gamma_proxy"], lags=32, test_type="rho", trend="c"))
print(PhillipsPerron(HEX["gamma_proxy"], lags=32, test_type="rho", trend="ct"))

print(PhillipsPerron(HEX["gk_vola"], lags=32, test_type="rho", trend="n")) #hits the 5% critical value!
print(PhillipsPerron(HEX["gk_vola"], lags=32, test_type="rho", trend="c"))
print(PhillipsPerron(HEX["gk_vola"], lags=32, test_type="rho", trend="ct"))

print(adfuller(HEX['sd_vola']))

###
#SMI: stationary.
print(PhillipsPerron(SMI["gamma_proxy"], lags=32, test_type="rho", trend="n"))
print(PhillipsPerron(SMI["gamma_proxy"], lags=32, test_type="rho", trend="c"))
print(PhillipsPerron(SMI["gamma_proxy"], lags=32, test_type="rho", trend="ct"))

print(PhillipsPerron(SMI["gk_vola"], lags=32, test_type="rho", trend="n"))
print(PhillipsPerron(SMI["gk_vola"], lags=32, test_type="rho", trend="c"))
print(PhillipsPerron(SMI["gk_vola"], lags=32, test_type="rho", trend="ct"))

###
#BEL20: 
print(PhillipsPerron(BEL20["gamma_proxy"], lags=32, test_type="rho", trend="n"))
print(PhillipsPerron(BEL20["gamma_proxy"], lags=32, test_type="rho", trend="c"))
print(PhillipsPerron(BEL20["gamma_proxy"], lags=32, test_type="rho", trend="ct"))

print(PhillipsPerron(BEL20["gk_vola"], lags=32, test_type="rho", trend="n"))
print(PhillipsPerron(BEL20["gk_vola"], lags=32, test_type="rho", trend="c"))
print(PhillipsPerron(BEL20["gk_vola"], lags=32, test_type="rho", trend="ct"))   



#%% Time series regression.

'''DAX model'''
#Nullmodel - DAX
#Using Newey-White standard errors (heteroscedasticity and autocorrelation robust.)
reg0=(smf.ols("gk_vola ~ gamma_proxy", data=DAX).fit().get_robustcov_results(cov_type="HAC",maxlags=32))
reg0.summary()

#Run for delta volatility (bc. of the potential unit root)
DAX['d_gk_vola']=DAX['gk_vola']-DAX['gk_vola'].shift(1)
DAX['d_gk_vola']=DAX['d_gk_vola'].fillna(method='bfill')
print(PhillipsPerron(DAX["d_gk_vola"], lags=32, test_type="rho", trend="n")) #stationary
reg1=(smf.ols("d_gk_vola ~ gamma_proxy", data=DAX).fit().get_robustcov_results(cov_type="HAC",maxlags=32))
reg1.summary()

#Lagged gamma proxy
DAX['gamma_lag']=DAX['gamma_proxy'].shift(1)
reg2=(smf.ols("d_gk_vola ~ gamma_lag", data=DAX).fit().get_robustcov_results(cov_type="HAC",maxlags=32))
reg2.summary()

#testing alternative historical volatility measures. 
DAX['d_sd_vola']=DAX['sd_vola']-DAX['sd_vola'].shift(1)
reg3=(smf.ols("d_sd_vola ~ gamma_proxy", data=DAX).fit().get_robustcov_results(cov_type="HAC",maxlags=32))
reg3.summary()

DAX['d_parkinson_vola']=DAX['parkinson_vola']-DAX['parkinson_vola'].shift(1)
reg4=(smf.ols("d_parkinson_vola ~ gamma_proxy", data=DAX).fit().get_robustcov_results(cov_type="HAC",maxlags=32))
reg4.summary()

reg5=(smf.ols("log_ret ~ gamma_proxy", data=DAX).fit().get_robustcov_results(cov_type="HAC",maxlags=32))
reg5.summary()

reg6=(smf.ols("log_ret ~ gamma_lag", data=DAX).fit().get_robustcov_results(cov_type="HAC",maxlags=32))
reg6.summary()

reg7=(smf.ols("d_gk_vola ~ gamma_proxy+gamma_lag", data=DAX).fit().get_robustcov_results(cov_type="HAC",maxlags=32))
reg7.summary()

#absolute log return
DAX['abs_log_ret']=np.abs(DAX['log_ret'])
print(adfuller(DAX['abs_log_ret']))
reg9=(smf.ols("abs_log_ret ~ gamma_lag", data=DAX).fit().get_robustcov_results(cov_type="HAC",maxlags=32))
reg9.summary()


#Visualisation with scatterplot
r_squared=reg9.rsquared
sns.set_style("white")
sns.set_palette("crest")
sns.lmplot(x="gamma_lag", y="abs_log_ret", scatter_kws={"alpha":0.1, "color":"#3297a8"}, data=DAX,line_kws={"color":"#c2185b"}, robust=False)
plt.title("DAX: Abszolút loghozamok vs. Γ(t-1)", size=16)
plt.annotate(f"R^2 = {r_squared:.2f}", xy=(0.7, 0.9), xycoords="axes fraction")
plt.show()

r_squared_2=reg1.rsquared
sns.set_style("white")
sns.set_palette("crest")
sns.lmplot(x="gamma_proxy", y="d_gk_vola", scatter_kws={"alpha":0.1, "color":"#3297a8"}, data=DAX,line_kws={"color":"#c2185b"}, robust=False)
plt.title("DAX: G-K volatilitás megváltozása vs. Γ", size=16)
plt.annotate(f"R^2 = {r_squared_2:.2f}", xy=(0.7, 0.9), xycoords="axes fraction")
plt.show()

###

#checking the residuals
residuals=reg4.resid
print(adfuller(residuals))
#Highest R^2: reg2 (gk_vola~gamma_lag): 7,4% 

#Regression table of the simple linear models
reg1=smf.ols("d_gk_vola ~ gamma_proxy", data=DAX).fit()
reg2=smf.ols("d_gk_vola ~ gamma_lag", data=DAX).fit()
reg3=smf.ols("d_sd_vola ~ gamma_proxy", data=DAX).fit()  
reg4=smf.ols("d_parkinson_vola ~ gamma_proxy", data=DAX).fit()
reg5=smf.ols("log_ret ~ gamma_proxy", data=DAX).fit()
reg6=smf.ols("log_ret ~ gamma_lag", data=DAX).fit()
reg7=smf.ols("d_gk_vola ~ gamma_proxy + gamma_lag", data=DAX).fit()      

stargazer = Stargazer([reg1,reg2,reg3,reg4,reg5,reg6, reg7])
stargazer.covariate_order(["gamma_proxy", "gamma_lag","Intercept"])
stargazer.rename_covariates(
    {
        "Intercept": "Constant",
        "gamma_proxy": "Γ",
        "gamma_lag":"Γ(t-1)"
    }
)
stargazer.custom_columns(
    [
        "d_gk_vola",
        "d_gk_vola",
        "d_sd_vola",
        "d_parkinson_vola",
        "log_ret",
        "log_ret",
        "d_gk_vola"
    ],
    [1, 1, 1, 1, 1,1,1],
)

html_code=stargazer.render_html() #Stargazer SE.-s should be fixed (HAC)!! 


#robustness test with control variables
DAX = DAX.rename(columns={'O/S': 'O_S'})

# corr=DAX[['gamma_proxy', 'VIX', 'Volume', 'ECB_RateHist', 'O_S', 'Fear Greed']].corr() #no confusing multicollinearity 
reg8=(smf.ols("d_gk_vola ~ gamma_proxy + EU_ESI + ECB_RateHist + VIX + Volume", data=DAX).fit().get_robustcov_results(cov_type="HAC",maxlags=32))
reg8.summary()

from sklearn.preprocessing import StandardScaler

DAX_scaled = DAX.copy()

# scale the variables fixing the large condition number
scaler = StandardScaler()
# DAX_scaled[['gamma_proxy', 'Fear Greed', 'ECB_RateHist', 'VIX', 'Volume', 'gamma_lag', 'O_S']] = scaler.fit_transform(DAX[['gamma_proxy', 'Fear Greed', 'ECB_RateHist', 'VIX', 'Volume', 'gamma_lag', 'O_S']])
DAX_scaled['d_gk_vola'] = scaler.fit_transform(DAX[['d_gk_vola']])
DAX_scaled['abs_log_ret'] = scaler.fit_transform(DAX[['abs_log_ret']])
# DAX_scaled=DAX_scaled.rename(columns={'Fear Greed':'Fear_Greed'})
# Fear and Greed could be included as well. 
DAX_scaled[['gamma_proxy', 'ECB_RateHist', 'VIX', 'Volume', 'gamma_lag', 'O_S']] = scaler.fit_transform(DAX[['gamma_proxy', 'ECB_RateHist', 'VIX', 'Volume', 'gamma_lag', 'O_S']])
reg10=(smf.ols("d_gk_vola ~ gamma_proxy +gamma_lag+ ECB_RateHist + VIX + Volume+O_S", data=DAX_scaled).fit().get_robustcov_results(cov_type="HAC",maxlags=32))
reg10.summary()
reg11=(smf.ols("abs_log_ret ~ gamma_proxy+gamma_lag+ ECB_RateHist + VIX + Volume+O_S", data=DAX_scaled).fit().get_robustcov_results(cov_type="HAC",maxlags=32))
reg11.summary()

# model_dax=(smf.ols("d_gk_vola ~ gamma_proxy +FearGreed + ECB_RateHist + VIX + Volume+O_S", data=DAX_scaled).fit().get_robustcov_results(cov_type="HAC",maxlags=32))
# model_dax.summary()#12,4% R^2, 0,00 p-value for gamma_proxy - robust & stabile


###

""" UKX model """
reg0=(smf.ols("gk_vola ~ gamma_proxy", data=UKX).fit().get_robustcov_results(cov_type="HAC",maxlags=32))
reg0.summary()

UKX['gamma_lag']=UKX['gamma_proxy'].shift(1)
UKX['abs_log_ret']=np.abs(UKX['log_ret'])
UKX['d_gk_vola']=UKX['gk_vola']-UKX['gk_vola'].shift(1)

reg1=(smf.ols("d_gk_vola ~ gamma_proxy", data=UKX).fit().get_robustcov_results(cov_type="HAC",maxlags=32))
reg1.summary()

reg2=(smf.ols("d_gk_vola ~ gamma_lag", data=UKX).fit().get_robustcov_results(cov_type="HAC",maxlags=32))
reg2.summary()

reg3=(smf.ols("log_ret ~ gamma_proxy", data=UKX).fit().get_robustcov_results(cov_type="HAC",maxlags=32))
reg3.summary()

reg4=(smf.ols("abs_log_ret ~ gamma_lag", data=UKX).fit().get_robustcov_results(cov_type="HAC",maxlags=32))
reg4.summary()

print(adfuller(UKX['abs_log_ret']))
reg5=(smf.ols("abs_log_ret ~ gamma_proxy+gamma_lag", data=UKX).fit().get_robustcov_results(cov_type="HAC",maxlags=32))
reg5.summary()


#Visualisation with scatterplot
sns.set_style("white")
r_squared=reg4.rsquared
fig, axs = plt.subplots(1, 2, figsize=(12, 6))
sns.regplot(ax=axs[0], x="gamma_lag", y="abs_log_ret", scatter_kws={"alpha":0.1, "color":"#3297a8"}, data=UKX,line_kws={"color":"#c2185b"})
axs[0].set_title("FTSE-100: Abszolút loghozamok vs. Γ(t-1)", size=16)
axs[0].annotate(f"R^2 = {r_squared:.2f}", xy=(0.7, 0.9), xycoords="axes fraction")

r_squared_2=reg1.rsquared
sns.regplot(ax=axs[1], x="gamma_proxy", y="d_gk_vola", scatter_kws={"alpha":0.1, "color":"#3297a8"}, data=UKX,line_kws={"color":"#c2185b"})
axs[1].set_title("FTSE-100: G-K volatilitás megváltozása vs. Γ", size=16)
axs[1].annotate(f"R^2 = {r_squared_2:.2f}", xy=(0.7, 0.9), xycoords="axes fraction")

plt.show()

###

reg1=smf.ols("d_gk_vola ~ gamma_proxy", data=UKX).fit()
reg2=smf.ols("d_gk_vola ~ gamma_lag", data=UKX).fit()
reg3=smf.ols("log_ret ~ gamma_proxy", data=UKX).fit()
reg4=smf.ols("log_ret ~ gamma_lag", data=UKX).fit()
reg5=smf.ols("abs_log_ret ~ gamma_proxy + gamma_lag", data=UKX).fit()

stargazer = Stargazer([reg1,reg2,reg3,reg4,reg5])
stargazer.covariate_order(["gamma_proxy", "gamma_lag","Intercept"])
stargazer.rename_covariates(
    {
        "Intercept": "Constant",
        "gamma_proxy": "Γ",
        "gamma_lag":"Γ(t-1)"
    }
)
stargazer.custom_columns(
    [
        "d_gk_vola",
        "d_gk_vola",
        "log_ret",
        "log_ret",
        "abs_logret"
    ],
    [1, 1, 1, 1, 1],
)

html_code=stargazer.render_html() 
###

#robust
UKX = UKX.rename(columns={'O/S': 'O_S'})
# UKX = UKX.rename(columns={'O/S': 'O_S', 'Fear Greed':'Fear_Greed'})
# corr=UKX[['gamma_proxy', 'Fear_Greed', 'VIX', 'Volume', 'BOE_Rate', 'O_S']].corr() #EU_ESI and VIX (?) valszeg le fog jönni -0,6 körülire 
UKX_scaled=UKX.copy()
UKX_scaled[['gk_vola', 'd_gk_vola', 'abs_log_ret']] = scaler.fit_transform(UKX[['gk_vola', 'd_gk_vola', 'abs_log_ret']])
# UKX_scaled[['gamma_proxy', 'Fear_Greed', 'BOE_Rate', 'VIX', 'Volume', 'gamma_lag', 'O_S']] = scaler.fit_transform(UKX[['gamma_proxy', 'Fear_Greed', 'BOE_Rate', 'VIX', 'Volume', 'gamma_lag', 'O_S']])
# UKX_scaled['parkinson_vola'] = scaler.fit_transform(UKX[['parkinson_vola']])
UKX_scaled[['gamma_proxy', 'BOE_Rate', 'VIX', 'Volume', 'gamma_lag', 'O_S']] = scaler.fit_transform(UKX[['gamma_proxy', 'BOE_Rate', 'VIX', 'Volume', 'gamma_lag', 'O_S']])
reg3=(smf.ols("d_gk_vola ~ gamma_proxy +gamma_lag+ BOE_Rate + VIX + Volume+O_S", data=UKX_scaled).fit().get_robustcov_results(cov_type="HAC",maxlags=32))
reg3.summary()
reg4=(smf.ols("abs_log_ret ~ gamma_proxy +gamma_lag+ BOE_Rate + VIX + Volume+O_S", data=UKX_scaled).fit().get_robustcov_results(cov_type="HAC",maxlags=32))
reg4.summary()

model_ukx=(smf.ols("d_gk_vola ~ gamma_proxy + EU_ESI + BOE_Rate + VIX + Volume+O_S", data=UKX_scaled).fit().get_robustcov_results(cov_type="HAC",maxlags=32))
model_ukx.summary()


###
""" SMI model """
reg0=(smf.ols("gk_vola ~ gamma_proxy", data=SMI).fit().get_robustcov_results(cov_type="HAC",maxlags=32))
reg0.summary()
SMI['d_gk_vola']=SMI['gk_vola']-SMI['gk_vola'].shift(1)
reg1=(smf.ols("d_gk_vola ~ gamma_proxy", data=SMI).fit().get_robustcov_results(cov_type="HAC",maxlags=32))
reg1.summary()
SMI['gamma_lag']=SMI['gamma_proxy'].shift(1)
reg2=(smf.ols("d_gk_vola ~ gamma_lag", data=SMI).fit().get_robustcov_results(cov_type="HAC",maxlags=32))
reg2.summary()
SMI['abs_log_ret']=np.abs(SMI['log_ret'])
reg3=(smf.ols("abs_log_ret ~ gamma_proxy", data=SMI).fit().get_robustcov_results(cov_type="HAC",maxlags=32))
reg3.summary()
reg4=(smf.ols("abs_log_ret ~ gamma_lag", data=SMI).fit().get_robustcov_results(cov_type="HAC",maxlags=32))
reg4.summary()
reg5=(smf.ols("abs_log_ret ~ gamma_proxy+gamma_lag", data=SMI).fit().get_robustcov_results(cov_type="HAC",maxlags=32))
reg5.summary()

#Visualisation with scatterplot
r_squared=reg3.rsquared
fig, axs = plt.subplots(1, 2, figsize=(12, 6))
sns.regplot(ax=axs[0], x="gamma_lag", y="abs_log_ret", scatter_kws={"alpha":0.1, "color":"#3297a8"}, data=SMI,line_kws={"color":"#c2185b"})
axs[0].set_title("SMI: Abszolút loghozamok vs. Γ(t-1)", size=16)
axs[0].annotate(f"R^2 = {r_squared:.2f}", xy=(0.7, 0.9), xycoords="axes fraction")

r_squared_2=reg1.rsquared
sns.regplot(ax=axs[1], x="gamma_proxy", y="d_gk_vola", scatter_kws={"alpha":0.1, "color":"#3297a8"}, data=SMI,line_kws={"color":"#c2185b"})
axs[1].set_title("SMI: G-K volatilitás megváltozása vs. Γ", size=16)
axs[1].annotate(f"R^2 = {r_squared_2:.2f}", xy=(0.7, 0.9), xycoords="axes fraction")

plt.show()

###
reg1=smf.ols("d_gk_vola ~ gamma_proxy", data=SMI).fit()
reg2=smf.ols("d_gk_vola ~ gamma_lag", data=SMI).fit()
reg3=smf.ols("log_ret ~ gamma_proxy", data=SMI).fit()
reg4=smf.ols("log_ret ~ gamma_lag", data=SMI).fit()
reg5=smf.ols("abs_log_ret ~ gamma_proxy+gamma_lag", data=SMI).fit()

stargazer = Stargazer([reg1,reg2,reg3,reg4,reg5])
stargazer.covariate_order(["gamma_proxy", "gamma_lag","Intercept"])
stargazer.rename_covariates(
    {
        "Intercept": "Constant",
        "gamma_proxy": "Γ",
        "gamma_lag":"Γ(t-1)"
    }
)
stargazer.custom_columns(
    [
        "d_gk_vola",
        "d_gk_vola",
        "log_ret",
        "log_ret",
        "abs_logret"
    ],
    [1, 1, 1, 1, 1],
)

html_code=stargazer.render_html() 

###

#robust - SNB rate history? 
SMI=SMI.rename(columns={'O/S':'O_S'})
# SMI=SMI.rename(columns={'O/S':'O_S', 'Fear Greed':'Fear_Greed'})
# SMI[['gamma_proxy', 'Fear_Greed', 'VIX', 'Volume', 'O_S']].corr()
SMI_scaled=SMI.copy()
# SMI_scaled[['gamma_proxy', 'Fear_Greed','VIX', 'Volume', 'gamma_lag', 'O_S']] = scaler.fit_transform(SMI[['gamma_proxy', 'Fear_Greed', 'VIX', 'Volume', 'gamma_lag', 'O_S']])
SMI_scaled[['gamma_proxy','VIX', 'Volume', 'gamma_lag', 'O_S']] = scaler.fit_transform(SMI[['gamma_proxy', 'VIX', 'Volume', 'gamma_lag', 'O_S']])
SMI_scaled[['abs_log_ret', 'd_gk_vola']] = scaler.fit_transform(SMI[['abs_log_ret', 'd_gk_vola']])

reg3=(smf.ols("d_gk_vola ~ gamma_proxy +gamma_lag + VIX + Volume + O_S ", data=SMI_scaled).fit().get_robustcov_results(cov_type="HAC",maxlags=32))
reg3.summary()
reg4=(smf.ols("abs_log_ret ~ gamma_proxy +gamma_lag + VIX + Volume + O_S ", data=SMI_scaled).fit().get_robustcov_results(cov_type="HAC",maxlags=32))
reg4.summary()

model_smi=(smf.ols("d_gk_vola ~ gamma_proxy + EU_ESI + VIX + Volume + O_S ", data=SMI_scaled).fit().get_robustcov_results(cov_type="HAC",maxlags=32))
model_smi.summary() #R^2=0.095, p-value=0.021
   
###
"""HEX"""
reg0=(smf.ols("gk_vola ~ gamma_proxy", data=HEX).fit().get_robustcov_results(cov_type="HAC",maxlags=32))
reg0.summary()
HEX['d_gk_vola']=HEX['gk_vola']-HEX['gk_vola'].shift(1)
HEX['gamma_lag']=HEX['gamma_proxy'].shift(1)
HEX['abs_log_ret']=np.abs(HEX['log_ret'])
reg1=(smf.ols("d_gk_vola ~ gamma_proxy", data=HEX).fit().get_robustcov_results(cov_type="HAC",maxlags=32))
reg1.summary()
reg2=(smf.ols("d_gk_vola ~ gamma_lag", data=HEX).fit().get_robustcov_results(cov_type="HAC",maxlags=32))
reg2.summary() #p-value=0.069* - weak
reg3=(smf.ols("log_ret ~ gamma_proxy", data=HEX).fit().get_robustcov_results(cov_type="HAC",maxlags=32))
reg3.summary()
reg4=(smf.ols("abs_log_ret ~ gamma_lag", data=HEX).fit().get_robustcov_results(cov_type="HAC",maxlags=32))
reg4.summary()
reg5=(smf.ols("abs_log_ret ~ gamma_proxy+gamma_lag", data=HEX).fit().get_robustcov_results(cov_type="HAC",maxlags=32))
reg5.summary()

#Visualisation with scatterplot
r_squared=reg4.rsquared
fig, axs = plt.subplots(1, 2, figsize=(12, 6))
sns.regplot(ax=axs[0], x="gamma_lag", y="abs_log_ret", scatter_kws={"alpha":0.1, "color":"#3297a8"}, data=HEX,line_kws={"color":"#c2185b"})
axs[0].set_title("HEX: Abszolút loghozamok vs. Γ(t-1)", size=16)
axs[0].annotate(f"R^2 = {r_squared:.2f}", xy=(0.7, 0.9), xycoords="axes fraction")

r_squared_2=reg1.rsquared
sns.regplot(ax=axs[1], x="gamma_proxy", y="d_gk_vola", scatter_kws={"alpha":0.1, "color":"#3297a8"}, data=HEX,line_kws={"color":"#c2185b"})
axs[1].set_title("HEX: G-K volatilitás megváltozása vs. Γ", size=16)
axs[1].annotate(f"R^2 = {r_squared_2:.2f}", xy=(0.7, 0.9), xycoords="axes fraction")

plt.show()

###
reg1=smf.ols("d_gk_vola ~ gamma_proxy", data=HEX).fit()
reg2=smf.ols("d_gk_vola ~ gamma_lag", data=HEX).fit()
reg3=smf.ols("log_ret ~ gamma_proxy", data=HEX).fit()
reg4=smf.ols("log_ret ~ gamma_lag", data=HEX).fit()
reg5=smf.ols("abs_log_ret ~ gamma_proxy+gamma_lag", data=HEX).fit()

stargazer = Stargazer([reg1,reg2,reg3,reg4,reg5])
stargazer.covariate_order(["gamma_proxy", "gamma_lag","Intercept"])
stargazer.rename_covariates(
    {
        "Intercept": "Constant",
        "gamma_proxy": "Γ",
        "gamma_lag":"Γ(t-1)"
    }
)
stargazer.custom_columns(
    [
        "d_gk_vola",
        "d_gk_vola",
        "log_ret",
        "log_ret",
        "abs_logret"
    ],
    [1, 1, 1, 1, 1],
)

html_code=stargazer.render_html() 

###
#robust 

"""HEX index: no significant connection!"""
HEX=HEX.rename(columns={'O/S':'O_S'})
# HEX=HEX.rename(columns={'O/S':'O_S', 'Fear Greed':'Fear_Greed'})
# HEX[['gamma_proxy', 'Fear_Greed', 'VIX', 'Volume', 'O_S', 'ECB_RateHist']].corr()
HEX_scaled=HEX.copy()
HEX_scaled[['gamma_proxy','VIX', 'Volume', 'gamma_lag', 'O_S','ECB_RateHist']] = scaler.fit_transform(HEX[['gamma_proxy', 'VIX', 'Volume', 'gamma_lag', 'O_S', 'ECB_RateHist']])
HEX_scaled[['gk_vola', 'd_gk_vola', 'abs_log_ret']] = scaler.fit_transform(HEX[['gk_vola', 'd_gk_vola', 'abs_log_ret']])

reg3=(smf.ols("d_gk_vola ~ gamma_proxy+gamma_lag+VIX+Volume+O_S+ECB_RateHist", data=HEX_scaled).fit().get_robustcov_results(cov_type="HAC",maxlags=32))
reg3.summary()
reg4=(smf.ols("abs_log_ret ~ gamma_proxy+gamma_lag+VIX+Volume+O_S+ECB_RateHist", data=HEX_scaled).fit().get_robustcov_results(cov_type="HAC",maxlags=32))
reg4.summary()

model_hex=(smf.ols("d_gk_vola ~ gamma_proxy+VIX+EU_ESI+Volume+O_S+ECB_RateHist", data=HEX_scaled).fit().get_robustcov_results(cov_type="HAC",maxlags=32))
model_hex.summary()

###
"""BEL20 -  no gamma values after 2017-09-15"""
BEL20 =BEL20[BEL20['Date'] <= '2017-09-15']
reg0=(smf.ols("gk_vola ~ gamma_proxy", data=BEL20).fit().get_robustcov_results(cov_type="HAC",maxlags=32))
reg0.summary()
BEL20['d_gk_vola']=BEL20['gk_vola']-BEL20['gk_vola'].shift(1)
BEL20['gamma_lag']=BEL20['gamma_proxy'].shift(1)
BEL20['abs_log_ret']=np.abs(BEL20['log_ret'])
reg1=(smf.ols("d_gk_vola ~ gamma_proxy", data=BEL20).fit().get_robustcov_results(cov_type="HAC",maxlags=32))
reg1.summary()
reg2=(smf.ols("d_gk_vola ~ gamma_lag", data=BEL20).fit().get_robustcov_results(cov_type="HAC",maxlags=32))
reg2.summary()
reg3=(smf.ols("log_ret ~ gamma_proxy", data=BEL20).fit().get_robustcov_results(cov_type="HAC",maxlags=32))
reg3.summary()
reg4=(smf.ols("abs_log_ret ~ gamma_lag", data=BEL20).fit().get_robustcov_results(cov_type="HAC",maxlags=32))
reg4.summary()
reg5=(smf.ols("abs_log_ret ~ gamma_proxy+ gamma_lag", data=BEL20).fit().get_robustcov_results(cov_type="HAC",maxlags=32))
reg5.summary()

#Visualisation with scatterplot
r_squared=reg4.rsquared
fig, axs = plt.subplots(1, 2, figsize=(12, 6))
sns.regplot(ax=axs[0], x="gamma_lag", y="abs_log_ret", scatter_kws={"alpha":0.1, "color":"#3297a8"}, data=BEL20,line_kws={"color":"#c2185b"})
axs[0].set_title("BEL20: Abszolút loghozamok vs. Γ(t-1)", size=16)
axs[0].annotate(f"R^2 = {r_squared:.2f}", xy=(0.7, 0.9), xycoords="axes fraction")

r_squared_2=reg1.rsquared
sns.regplot(ax=axs[1], x="gamma_proxy", y="d_gk_vola", scatter_kws={"alpha":0.1, "color":"#3297a8"}, data=BEL20,line_kws={"color":"#c2185b"})
axs[1].set_title("BEL20: G-K volatilitás megváltozása vs. Γ", size=16)
axs[1].annotate(f"R^2 = {r_squared_2:.2f}", xy=(0.7, 0.9), xycoords="axes fraction")

plt.show()

###

reg1=smf.ols("d_gk_vola ~ gamma_proxy", data=BEL20).fit()
reg2=smf.ols("d_gk_vola ~ gamma_lag", data=BEL20).fit()
reg3=smf.ols("log_ret ~ gamma_proxy", data=BEL20).fit()
reg4=smf.ols("log_ret ~ gamma_lag", data=BEL20).fit()
reg5=smf.ols("abs_log_ret ~ gamma_proxy+gamma_lag", data=BEL20).fit()

stargazer = Stargazer([reg1,reg2,reg3,reg4,reg5])
stargazer.covariate_order(["gamma_proxy", "gamma_lag","Intercept"])
stargazer.rename_covariates(
    {
        "Intercept": "Constant",
        "gamma_proxy": "Γ",
        "gamma_lag":"Γ(t-1)"
    }
)
stargazer.custom_columns(
    [
        "d_gk_vola",
        "d_gk_vola",
        "log_ret",
        "log_ret",
        "abs_logret"
    ],
    [1, 1, 1, 1, 1],
)

html_code=stargazer.render_html() 

###

BEL20=BEL20.rename(columns={'O/S':'O_S'})
# BEL20=BEL20.rename(columns={'O/S':'O_S', 'Fear Greed':'Fear_Greed'})
BEL20[['gamma_proxy', 'VIX', 'Volume', 'O_S', 'ECB_RateHist']].corr()
BEL20_scaled=BEL20.copy()
BEL20_scaled[['gamma_proxy', 'VIX', 'Volume', 'gamma_lag', 'O_S','ECB_RateHist']] = scaler.fit_transform(BEL20[['gamma_proxy', 'VIX', 'Volume', 'gamma_lag', 'O_S', 'ECB_RateHist']])
BEL20_scaled[['gk_vola', 'd_gk_vola', 'abs_log_ret']] = scaler.fit_transform(BEL20[['gk_vola', 'd_gk_vola', 'abs_log_ret']])
reg3=(smf.ols("d_gk_vola ~ gamma_proxy+gamma_lag+VIX+Volume+O_S+ECB_RateHist", data=BEL20_scaled).fit().get_robustcov_results(cov_type="HAC",maxlags=32))
reg3.summary()
reg4=(smf.ols("abs_log_ret ~ gamma_proxy+gamma_lag+VIX+Volume+O_S+ECB_RateHist", data=BEL20_scaled).fit().get_robustcov_results(cov_type="HAC",maxlags=32))
reg4.summary()
model_bel20=(smf.ols("d_gk_vola ~ gamma_proxy+VIX+EU_ESI+Volume+O_S+ECB_RateHist", data=BEL20_scaled).fit().get_robustcov_results(cov_type="HAC",maxlags=32))
model_bel20.summary()

###
# model_dax=smf.ols("abs_log_ret ~ gamma_lag+Fear_Greed + ECB_RateHist + VIX + Volume+O_S", data=DAX_scaled).fit()
# model_ukx=smf.ols("abs_log_ret ~ gamma_proxy + gamma_lag+Fear_Greed + BOE_Rate + VIX + Volume+O_S", data=UKX_scaled).fit()
# model_smi=smf.ols("abs_log_ret ~ gamma_proxy +gamma_lag+ Fear_Greed + VIX + Volume + O_S ", data=SMI_scaled).fit()
# model_hex=smf.ols("abs_log_ret ~ gamma_proxy+VIX+Fear_Greed ++Volume+O_S+ECB_RateHist", data=HEX_scaled).fit()
# model_bel20=smf.ols("abs_log_ret ~ gamma_proxy+VIX+Fear_Greed ++Volume+O_S+ECB_RateHist", data=BEL20_scaled).fit()

model_dax=smf.ols("d_gk_vola ~ gamma_proxy+gamma_lag+ ECB_RateHist + VIX + Volume+O_S", data=DAX_scaled).fit()
model_ukx=smf.ols("d_gk_vola ~ gamma_proxy + gamma_lag+ BOE_Rate + VIX + Volume+O_S", data=UKX_scaled).fit()
model_smi=smf.ols("d_gk_vola ~ gamma_proxy +gamma_lag+ VIX + Volume + O_S ", data=SMI_scaled).fit()
model_hex=smf.ols("d_gk_vola ~ gamma_proxy+gamma_lag+VIX+Volume+O_S+ECB_RateHist", data=HEX_scaled).fit()
model_bel20=smf.ols("d_gk_vola ~ gamma_proxy+gamma_lag+VIX+Volume+O_S+ECB_RateHist", data=BEL20_scaled).fit()

stargazer = Stargazer([model_dax, model_ukx, model_smi, model_hex, model_bel20])
stargazer.covariate_order(["gamma_proxy", "gamma_lag", "VIX", "Volume", "O_S", "ECB_RateHist", "BOE_Rate", "Intercept"])
stargazer.rename_covariates(
    {
        "Intercept": "Constant",
        "gamma_proxy": "Γ",
        "gamma_lag": "Γ(t-1)",
        "VIX":"VIX",
        "Volume":"Volume",
        "O_S":"O/S",
        "ECB_RateHist":"ECB rate change",
        "BOE_Rate":"BoE rate change"
    }
)
stargazer.custom_columns(
    [
        "DAX",
        "FTSE-100",
        "SMI",
        "HEX",
        "BEL20"
    ],
    [1, 1, 1, 1, 1],
)

html_code=stargazer.render_html() #view with: https://codebeautify.org/htmlviewer#

del BEL20_scaled, HEX_scaled, reg0, reg1, reg2, reg3, reg4, reg5,reg6,reg7,reg8,reg9, reg 10, scaler, SMI_scaled, stargazer, UKX_scaled
del r_squared, r_squared_2, fig, axs, reg11, residuals

#%% Quantile regression - not included in the final version.
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
import statsmodels.api as sm

# define the dependent and explanatory variables
y = DAX_scaled['abs_log_ret']
X = DAX_scaled[['gamma_proxy', 'VIX', 'O_S', 'Volume', 'EU_ESI', 'ECB_RateHist']]

# add a constant term to the explanatory variables
X = sm.add_constant(X)

# specify the quantile levels for the regression
quantiles = [0.1, 0.5, 0.9]

# fit the quantile regression models for the specified quantiles
models = []
for q in quantiles:
    model = sm.QuantReg(y, X).fit(q=q)
    models.append(model)

# print the summary of each quantile regression model
for i, model in enumerate(models):
    print(f"Quantile regression results for {quantiles[i]}:")
    print(model.summary())
    

del DAX_scaled, i, y, X, q
#%% Option to stock volume chart etc.

# DAX_avg = DAX.groupby(pd.Grouper(key='Date', freq='Y')).mean().reset_index()
# DAX_avg['Ticker'] = 'DAX'
# HEX_avg = HEX.groupby(pd.Grouper(key='Date', freq='Y')).mean().reset_index()
# HEX_avg['Ticker'] = 'HEX'
# SMI_avg = SMI.groupby(pd.Grouper(key='Date', freq='Y')).mean().reset_index()
# SMI_avg['Ticker'] = 'SMI'
# BEL20_avg = BEL20.groupby(pd.Grouper(key='Date', freq='Y')).mean().reset_index()
# BEL20_avg['Ticker'] = 'BEL20'
# UKX_avg = UKX.groupby(pd.Grouper(key='Date', freq='Y')).mean().reset_index()
# UKX_avg['Ticker'] = 'UKX'

# df = pd.concat([DAX_avg, HEX_avg, SMI_avg, BEL20_avg, UKX_avg], ignore_index=True)

# df = df.rename(columns={'O_S': 'O_S_avg', 'OptVolume': 'OptVolume_avg'})

# df = df[['Ticker', 'Date', 'O_S_avg', 'OptVolume_avg']]
# df['Date'] = pd.to_datetime(df['Date'], format='%Y')
# df['Date'] = df['Date'].dt.strftime('%Y')
# df['Ticker'].replace({'UKX': 'FTSE100'}, inplace=True)
# df_pivot = df.pivot(index='Date', columns='Ticker', values='O_S_avg')

# ratio=pd.DataFrame(columns=['Date', 'ratio'])
# ratio['ratio']= df.loc[df['Ticker']=='UKX', 'OptVolume_avg'].values / df.loc[df['Ticker']=='SMI', 'OptVolume_avg'].values
# ratio['Date']=pd.date_range(start='2002', end='2020', freq='Y').year.astype(str)
# ratio['Date'] = ratio['Date'].dt.strftime('%Y')

# sns.set_style('white')
# sns.set_palette(['#2222a8', '#3297a8','#00ffff', '#555555', '#c2185b'])

# totals = df_pivot.sum(axis=1)
# df_perc = df_pivot.divide(totals, axis=0)

# ax = df_perc.plot(kind='bar', stacked=True, figsize=(12, 8))

# ax.set_title('Átlagos option to stock arány és átlagos opciós volumen relatív megoszlása ', fontsize=16)
# ax.set_xlabel('Év', fontsize=14)
# ax.set_ylabel('%', fontsize=14)

# vals = ax.get_yticks()
# ax.set_yticklabels(['{:,.0%}'.format(x) for x in vals])

# ax.legend(title='Ticker', fontsize=12, bbox_to_anchor=(1.05, 1))
# ax2 = ax.twinx()
# ax2.plot(ratio['Date'], ratio['ratio'], color='black', linewidth=5, linestyle="dotted")
# ax2.set_ylabel('Opciós volumen FTSE-100/SMI', fontsize=14)
# ax2.tick_params(axis='y', labelsize=12)

# plt.show()

# del ax, ax2, BEL20_avg, SMI_avg, DAX_scaled, DAX_avg, df_perc, df_pivot, HEX_avg, i, models, q, quantiles, ratio, totals, UKX_avg, vals, X , y


# df_merged = pd.merge(DAX[['Date', 'Volume', 'OptVolume']], UKX[['Date', 'Volume', 'OptVolume']], on='Date', suffixes=('_DAX', '_UKX'))
# df_merged = df_merged[df_merged['OptVolume_UKX'] > 5000]
# df_merged = df_merged[df_merged['OptVolume_DAX'] > 5000]
# df_merged['ratio_V']=df_merged['Volume_UKX']/df_merged['Volume_DAX']
# df_merged['ratio_V'].describe().T
# df_merged['ratio_V']=df_merged['OptVolume_UKX']/df_merged['OptVolume_DAX']
# df_merged['ratio_V'].describe()

# ###Some more plots

# quant_10 = DAX['gamma_proxy'].quantile(.1)
# quant_25 = DAX['gamma_proxy'].quantile(.25)
# median = DAX['gamma_proxy'].median()
# quant_75 = DAX['gamma_proxy'].quantile(.75)
# quant_90 = DAX['gamma_proxy'].quantile(.9)

# ax = sns.lineplot(data=DAX, x=DAX.index, y='gamma_proxy', color="#3297a8", linewidth=2)

# ax.axhline(median, color="teal", lw=3, linestyle='--')

# ax.fill_between(DAX.index, y1=quant_25, y2=quant_75, alpha=.3, color='teal')

# ax.fill_between(DAX.index, y1=quant_10, y2=quant_90, alpha=.1, color='teal')

# ax.set_xlabel('')
# ax.set_ylabel('Nettó gamma kitettség')

# ax.set_title('Keresztmetszeti eloszlása a gamma proxy-nak', size=16)

# sns.despine()

# plt.show()
# del df_merged, quant_10, quant_25, median, quant_75, quant_90

# ###some more descriptives

# DAX_new=DAX[DAX['Date'] >= '2011-01-01']
# DAX_new[['abs_log_ret','d_gk_vola']].describe()

# UKX_new=UKX[UKX['Date'] >= '2011-01-01']
# UKX_new[['abs_log_ret','d_gk_vola']].describe()

# SMI_new=SMI[SMI['Date'] >= '2011-01-01']
# SMI_new[['abs_log_ret','d_gk_vola']].describe()
# del DAX_new, UKX_new, SMI_new
