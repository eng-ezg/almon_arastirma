import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pandas_datareader import data as web
import statsmodels.api as sm


start = "1990-01-01"
end = "2024-12-31"

cpi = web.DataReader("CPIAUCSL", "fred", start, end)
rate = web.DataReader("FEDFUNDS", "fred", start, end)

inflation = 100 * np.log(cpi).diff()

df = pd.concat([inflation, rate], axis=1)
df.columns = ["inflation","interest"]

df = df.dropna()

lag = 12

for i in range(lag+1):
    df[f"r{i}"] = df["interest"].shift(i)

df = df.dropna()

z0 = np.zeros(len(df))
z1 = np.zeros(len(df))
z2 = np.zeros(len(df))

for i in range(lag+1):
    x = df[f"r{i}"]
    z0 += x
    z1 += i*x
    z2 += (i**2)*x

df["z0"] = z0
df["z1"] = z1
df["z2"] = z2

X = sm.add_constant(df[["z0","z1","z2"]])
y = df["inflation"]

model = sm.OLS(y, X).fit()

print(model.summary())

a0 = model.params["z0"]
a1 = model.params["z1"]
a2 = model.params["z2"]

lags = np.arange(lag+1)

beta = a0 + a1*lags + a2*(lags**2)

print(beta)

plt.figure(figsize=(8,4))

plt.plot(lags, beta, marker="o")

plt.xlabel("Lag (Ay)")
plt.ylabel("Etki")
plt.title("Almon Lag Katsayıları")
plt.show()
