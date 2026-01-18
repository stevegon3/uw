import featuretools as ft
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

if __name__ == "__main__":
    df = pd.read_excel("daily.xlsx", parse_dates=['date']).dropna()
    obj_cols = df.select_dtypes(include=['object']).columns
    df[obj_cols] = df[obj_cols].astype(float)
    es = ft.EntitySet(id="stock_info")
    es = es.add_dataframe(dataframe_name='daily', dataframe=df, index='id', make_index=True)
    f_matrix, f_defs = ft.dfs(entityset=es, target_dataframe_name="daily",
                              trans_primitives=['time_since', 'day', 'is_weekend', 'cum_min', 'minute',
                                                'num_words', 'weekday', 'cum_count', 'percentile', 'year', 'week', 'cum_mean'])

    # List the generated features
    print(list(f_matrix))

    # Run a simple regression so we can see feature importance
    # Remove gspc, DJIA from features
    features = []
    for f in f_matrix.columns:
        if 'gspc' not in f.lower() and 'dji' not in f.lower():
            features.append(f)
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=8,
        random_state=42
    )
    X_train, X_test, y_train, y_test = train_test_split(f_matrix[features], df['gspc'], test_size=0.2, shuffle=False)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print("Mean Squared Error:", mse)
    print("R^2 Score:", r2)
    # from the calculated importances, order them from most to least important
    # and make a barplot so we can visualize what is/isn't important
    plt.figure(figsize=(12, 15))
    importances = model.feature_importances_
    sorted_idx = np.argsort(importances)[-15:]  # indices of top 15

    # Prepare labels and values
    top_features = np.array(features)[sorted_idx]
    top_importances = importances[sorted_idx]
    padding = np.arange(len(top_features)) + 1

    plt.figure(figsize=(10, 12))
    plt.barh(padding, top_importances, align='center')
    plt.yticks(padding, top_features)
    plt.xlabel("Relative Importance")
    plt.title("Top 15 Variable Importances")
    plt.show()