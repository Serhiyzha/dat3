import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-SkillsNetwork/labs/Data%20files/auto.csv"

headers = ["symboling", "normalized-losses", "make", "fuel-type", "aspiration", 
           "num-of-doors", "body-style", "drive-wheels", "engine-location", "wheel-base", 
           "length", "width", "height", "curb-weight", "engine-type", 
           "num-of-cylinders", "engine-size", "fuel-system", "bore", "stroke", 
           "compression-ratio", "horsepower", "peak-rpm", "city-mpg", "highway-mpg", "price"]

df = pd.read_csv(url, header=None, names=headers)
df.replace("?", np.nan, inplace=True)

numeric_cols = ["engine-size", "price", "normalized-losses", "highway-mpg", "horsepower", "curb-weight"]
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col])

df.dropna(subset=["price"], inplace=True)
df["normalized-losses"] = df["normalized-losses"].fillna(df["normalized-losses"].mean())
df["horsepower"] = df["horsepower"].fillna(df["horsepower"].mean())

Y = df["price"]

lm1 = LinearRegression()
X1 = df[["engine-size"]]
lm1.fit(X1, Y)
Yhat1 = lm1.predict(X1)

print("Проста лінійна регресія (engine-size -> price):")
print(f"R^2: {r2_score(Y, Yhat1):.4f}, MSE: {mean_squared_error(Y, Yhat1):.2f}")

lm2 = LinearRegression()
Z = df[["normalized-losses", "highway-mpg"]]
lm2.fit(Z, Y)
Yhat2 = lm2.predict(Z)

print("\nМножинна лінійна регресія (normalized-losses, highway-mpg -> price):")
print(f"R^2: {r2_score(Y, Yhat2):.4f}, MSE: {mean_squared_error(Y, Yhat2):.2f}")

x_poly = df["highway-mpg"]
p11 = np.poly1d(np.polyfit(x_poly, Y, 11))
Yhat3 = p11(x_poly)

print("\nПоліноміальна модель 11-го порядку (highway-mpg -> price):")
print(f"R^2: {r2_score(Y, Yhat3):.4f}, MSE: {mean_squared_error(Y, Yhat3):.2f}")

X4 = df[["horsepower"]]
x4_flat = df["horsepower"]

m_lin = LinearRegression().fit(X4, Y)
pred_lin = m_lin.predict(X4)

p3 = np.poly1d(np.polyfit(x4_flat, Y, 3))
pred_poly3 = p3(x4_flat)

Z4 = df[["horsepower", "engine-size", "curb-weight"]]
m_multi = LinearRegression().fit(Z4, Y)
pred_multi = m_multi.predict(Z4)

print("\nПорівняння моделей (для price):")
print(f"Проста лінійна (horsepower):       R^2 = {r2_score(Y, pred_lin):.4f}, MSE = {mean_squared_error(Y, pred_lin):.2f}")
print(f"Поліноміальна 3-го порядку (hp):   R^2 = {r2_score(Y, pred_poly3):.4f}, MSE = {mean_squared_error(Y, pred_poly3):.2f}")
print(f"Множинна лінійна (3 ознаки):       R^2 = {r2_score(Y, pred_multi):.4f}, MSE = {mean_squared_error(Y, pred_multi):.2f}")