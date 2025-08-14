import pandas as pd
import numpy as np
pd.set_option('display.max_columns', None)

def scrub_col(col):
    return col.lower().replace(" ", "_").replace('^', '').replace('_price', '').strip()

def convert_to_float(df, ignore_cols=['date']):
    # Takes DF and updates it in place
    obj_cols = monthly_prices.select_dtypes(include=['object']).columns
    obj_cols = [x for x in obj_cols if x not in ignore_cols]
    if obj_cols:
        df[obj_cols] = df[obj_cols].astype(float)

if __name__ == "__main__":
    daily_df = pd.read_excel("FMP.xlsx", sheet_name="FMP Daily", parse_dates=['date'])
    monthly_df = pd.read_excel("FMP.xlsx", sheet_name="FRED Mo", parse_dates=['date'])
    quarterly_df = pd.read_excel("FMP.xlsx", sheet_name="FRED Qtr", parse_dates=['date'])

    daily_df = daily_df.sort_values('date').set_index('date')
    monthly_df = monthly_df.sort_values('date')
    quarterly_df = quarterly_df.sort_values('date')

    daily_df.columns = [scrub_col(col) for col in daily_df.columns]
    monthly_df.columns = [scrub_col(col) for col in monthly_df.columns]
    quarterly_df.columns = [scrub_col(col) for col in quarterly_df.columns]

    daily_cols = [col for col in daily_df.columns if col != 'date' and col != '']  # SP500, DJIA, your stock
    monthly_cols = [col for col in monthly_df.columns if col != 'date' and col != '']  # Labor, Imports...
    quarterly_cols = [col for col in quarterly_df.columns if col != 'date' and col != '']  # GDP

    gdp_monthly = quarterly_df.set_index('date').resample('QE').last()
    gdp_monthly = gdp_monthly.resample('ME').ffill()
    macro_monthly = monthly_df.set_index('date').resample('ME').last()
    all_monthly = pd.concat([macro_monthly, gdp_monthly], axis=1)
    all_monthly_daily = all_monthly.resample('D').ffill()
    daily_df_all = daily_df.join([all_monthly_daily], how='left')
    print(daily_df_all.tail(3))
    with pd.ExcelWriter('daily.xlsx') as writer:
        daily_df_all.to_excel(writer, sheet_name='daily')