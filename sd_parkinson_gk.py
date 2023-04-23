#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 23 13:37:16 2023

@author: horvathbalint
"""

import pandas as pd
import numpy as np
import math
import yfinance as yf
import seaborn as sns
import matplotlib.pyplot as plt

"""Computing alternative historical volatilities."""
data=yf.download("^GDAXI", start="2001-01-01", end="2020-01-01")
data['Close'].plot()

#Standard deviation
def standard_deviation(price_data, window=30, trading_periods=252, clean=True):

    log_return = (price_data["Close"] / price_data["Close"].shift(1)).apply(np.log)

    result = log_return.rolling(window=window, center=False).std() * math.sqrt(
        trading_periods
    )

    if clean:
        return result.dropna()
    else:
        return result

standard_deviation(data).plot()

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
    
parkinson(data).plot()

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

garman_klass(data).plot()
   
gk_vol = garman_klass(data)
pk_vol = parkinson(data)
sd_vol = standard_deviation(data)

vol_data = pd.DataFrame({'Garman-Klass': gk_vol, 'Parkinson': pk_vol, 'Close-to-Close': sd_vol})

sns.set_palette("crest")
sns.set_style("whitegrid")

fig, ax = plt.subplots(figsize=(12, 8))
sns.lineplot(data=vol_data, ax=ax, palette=['#3297a8', '#c2185b', '#7b9095'])
ax.set_xlabel('Dátum', size=16)
ax.set_ylabel('Realizált volatilitás', size=16)
ax.set_title('Volatilitás becslő eljárások összevetése', size=22)
ax.legend(fontsize=16)
plt.show()