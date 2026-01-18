import pandas as pd
import numpy as np

df = pd.read_excel('Assignment-2_Problem-2_kangaroo.xls', engine='xlrd')
print(df.shape)
df = df[['X', 'Y']].copy()
df = df.apply(pd.to_numeric, errors='coerce')
df = df.dropna()
print('After dropna', df.shape)

x = df['X'].values
y = df['Y'].values
x_mean, x_std = x.mean(), x.std()
if x_std == 0: x_std = 1.0
y_mean = y.mean()
Xc = (x - x_mean) / x_std
print('x', x)
print('Xc', Xc)
yc = y - y_mean
m = 0.0
b = 0.0
learning_rate = 0.01
epochs = 100000
epochs = 3000
n = len(Xc)

for i in range(epochs):
    y_pred = m * Xc + b
    error = y_pred - yc
    dm = (2/n) * np.dot(error, Xc)
    db = (2/n) * np.sum(error)
    m -= learning_rate * dm
    b -= learning_rate * db
    if i % 200 == 0:
        print(f"Epoch {i}: m={m:.4f}, b={b:.4f}")

m = m / x_std
b = (b + y_mean) - m * x_mean
print(f"Optimal slope (m): {m:.4f}")
print(f"Optimal intercept (b): {b:.4f}")