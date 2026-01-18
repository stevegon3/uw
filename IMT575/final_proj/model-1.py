import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor

df = pd.read_excel("data_monthly.xlsx", parse_dates=['date']).dropna()
with pd.ExcelWriter('data_monthly_filtered.xlsx') as writer:
    df.to_excel(writer)
obj_cols = df.select_dtypes(include=['object']).columns
df[obj_cols] = df[obj_cols].astype(float)
print(df.dtypes)
print(df.head(5))
cols = ['date', 'aapl_ret', 'adm_ret', 'amzn_ret', 'avgo_ret', 'brk-b_ret',
       'f_ret', 'fcx_ret', 'gm_ret', 'googl_ret', 'meta_ret', 'msft_ret',
       'nke_ret', 'nvda_ret', 'tsla_ret', 'tsm_ret', 'wmt_ret', 'dji_ret',
       'gspc_ret', 'population', 'women_lpr', 'men_lpr', 'civilian_lpr',
       'civilian_ur', 'impca', 'impch', 'impfr', 'impge', 'impjp', 'impmx',
       'impkr', 'impuk', 'gdp']
x_cols = [x for x in cols if x not in ['date', 'gspc']]
X, y = df[x_cols], df['gspc']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, shuffle=False)
if False:
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=8,
        random_state=42
    )
    model = MLPRegressor(
        hidden_layer_sizes=(64, 32),
        activation='relu',
        solver='adam',
        max_iter=500,
        random_state=42
    )

model = XGBRegressor(
    n_estimators=100,
    max_depth=8000,
    learning_rate=10,
    random_state=42
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Mean Squared Error:", mse)
print("R^2 Score:", r2)