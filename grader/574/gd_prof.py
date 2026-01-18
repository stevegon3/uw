import pandas as pd
import numpy as np

df = pd.read_excel('Assignment-2_Problem-2_kangaroo.xls', engine='xlrd')

x = df['X'].values
y = df['Y'].values
print(x.shape)
n = len(x)
x_mean, x_std = x.mean(), x.std()
if x_std == 0: x_std = 1.0
y_mean = y.mean()
Xc = (x - x_mean) / x_std
print('x', x)
print('Xc', Xc)
yc = y - y_mean
m_c = 0.0
b_c = 0.0
learning_rate = .01  # 1e-2
epochs = 3000
for i in range(epochs):
    y_pred = m_c * Xc + b_c
    error  = y_pred - yc
    dm = (2.0/n) * np.dot(error, Xc)
    db = (2.0/n) * np.sum(error)

    m_c -= learning_rate * dm
    b_c -= learning_rate * db

    if i % 200 == 0:
        mse_c = float(np.mean((yc - y_pred)**2))
        print(f"Epoch {i:5d}: m_c={m_c: .6f}, b_c={b_c: .6f}, MSE_centered={mse_c: .6f}")
m = m_c / x_std
b = (b_c + y_mean) - m * x_mean

y_hat = m * x + b

print(f"Optimal slope (m): {m:.9f}")
print(f"Optimal intercept (b): {b:.9f}")