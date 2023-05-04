import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from math import sqrt, exp
from mpl_toolkits.mplot3d import Axes3D

class BS:

    def __init__(self, call, stock, strike, maturity, interest, volatility, dividend):
        self.call = call
        self.stock = stock
        self.strike = strike
        self.maturity = maturity
        self.interest = interest
        self.volatility = volatility
        self.dividend = dividend
        self.d1 = (self.volatility * sqrt(self.maturity)) ** (-1) * (
        np.log(self.stock / self.strike) + (self.interest - self.dividend + self.volatility ** 2 / 2) * self.maturity)
        self.d2 = self.d1 - self.volatility * sqrt(self.maturity)

    def price(self):
        if self.call:
            return exp(-self.dividend * self.maturity) * norm.cdf(self.d1) * self.stock - norm.cdf(
                self.d2) * self.strike * exp(-self.interest * self.maturity)
        else:
            return norm.cdf(-self.d2) * self.strike * exp(-self.interest * self.maturity) - norm.cdf(
                -self.d1) * self.stock * exp(-self.dividend * self.maturity)

    def delta(self):
        if self.call:
            return norm.cdf(self.d1) * exp(-self.dividend * self.maturity)
        else:
            return (norm.cdf(self.d1) - 1) * exp(-self.dividend * self.maturity)

    def gamma(self):
        return exp(-self.dividend * self.maturity) * norm.pdf(self.d1) / (
        self.stock * self.volatility * sqrt(self.maturity))

S = np.linspace(1, 50, 50)
T = np.linspace(0.01, 3, 50)

# Calculate gamma for different stock price and time to maturity
g = np.array([])
for i in range(0, len(T)):
    g = np.append(g, BS(True, S, 25, T[i], 0.15, 0.3, 0.00).gamma(), axis=0)
g = g.reshape(len(S), len(T))

X, Y = np.meshgrid(S, T)

sns.set_palette("magma")

fig = plt.figure(figsize=(14,14))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(Y, X, g, rstride=1, cstride=1, cmap='crest', shade=0.5)
ax.set_title('Gamma alakulása a lejárat és a moneyness függvényében', size=24)
ax.set_xlabel('Lejárat', size=20)
ax.set_ylabel('Részvény árfolyam', size=20)
ax.set_zlabel('Gamma', size=20)

# Add a line for the strike price
ax.plot(T, [25]*len(T), g[:,-1], color='red', linewidth=2, label='Strike Price') 

plt.show()
