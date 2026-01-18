import pandas as pd
import numpy as np
pd.set_option('display.max_columns', None)

def scrub_col(col):
    return col.lower().replace(" ", "_").replace('^', '').replace('_price', '').strip()

def create_leads(df, columns, max_lead=6):
    lead_df = pd.DataFrame(index=df.index)
    for col in columns:
        for lead in range(1, max_lead + 1):
            lead_df[f"{col}_lead{lead}"] = df[col].shift(lead)
    return lead_df

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
    daily_df = daily_df.sort_values('date')
    monthly_df = monthly_df.sort_values('date')
    quarterly_df = quarterly_df.sort_values('date')

    daily_df.columns = [scrub_col(col) for col in daily_df.columns]
    monthly_df.columns = [scrub_col(col) for col in monthly_df.columns]
    quarterly_df.columns = [scrub_col(col) for col in quarterly_df.columns]

    daily_cols = [col for col in daily_df.columns if col != 'date' and col != '']  # SP500, DJIA, your stock
    monthly_cols = [col for col in monthly_df.columns if col != 'date' and col != '']  # Labor, Imports...
    quarterly_cols = [col for col in quarterly_df.columns if col != 'date' and col != '']  # GDP

    monthly_prices = daily_df.set_index('date')[daily_cols].resample('ME').last()
    # Calculate monthly log returns
    monthly_returns = np.log(monthly_prices / monthly_prices.shift(1))
    monthly_returns.columns = [f"{col}_ret" for col in monthly_returns.columns]
    macro_monthly = monthly_df.set_index('date').resample('ME').last()
    # Quarterly GDP → convert to monthly with forward fill
    gdp_monthly = quarterly_df.set_index('date').resample('QE').last()
    gdp_monthly = gdp_monthly.resample('ME').ffill()
    # Combine everything
    data = pd.concat([monthly_returns, macro_monthly, gdp_monthly], axis=1)

    daily_df = pd.read_excel("FMP.xlsx", sheet_name="FMP Daily", parse_dates=['date'])
    monthly_df = pd.read_excel("FMP.xlsx", sheet_name="FRED Mo", parse_dates=['date'])
    quarterly_df = pd.read_excel("FMP.xlsx", sheet_name="FRED Qtr", parse_dates=['date'])
    daily_df = daily_df.sort_values('date')
    monthly_df = monthly_df.sort_values('date')
    quarterly_df = quarterly_df.sort_values('date')

    daily_df.columns = [scrub_col(col) for col in daily_df.columns]
    monthly_df.columns = [scrub_col(col) for col in monthly_df.columns]
    quarterly_df.columns = [scrub_col(col) for col in quarterly_df.columns]

    daily_cols = [scrub_col(col) for col in daily_df.columns if col != 'date' and col != '']  # SP500, DJIA, your stock
    monthly_cols = [col for col in monthly_df.columns if col != 'date' and col != '']  # Labor, Imports...
    quarterly_cols = [col for col in quarterly_df.columns if col != 'date' and col != '']  # GDP

    monthly_prices = daily_df.set_index('date')[daily_cols].resample('ME').last()
    convert_to_float(monthly_prices)
    print(monthly_prices.columns)
    with pd.ExcelWriter('prices_monthly.xlsx') as writer:
        monthly_prices.to_excel(writer, sheet_name='monthly')
    # Calculate monthly log returns
    monthly_returns = np.log(monthly_prices / monthly_prices.shift(1))
    monthly_returns.columns = [f"{col}_ret" for col in monthly_returns.columns]
    macro_monthly = monthly_df.set_index('date').resample('ME').last()
    # Quarterly GDP → convert to monthly with forward fill
    gdp_monthly = quarterly_df.set_index('date').resample('QE').last()
    gdp_monthly = gdp_monthly.resample('ME').ffill()

    # Combine everything as monthly
    data_mo = pd.concat([monthly_returns, macro_monthly, gdp_monthly], axis=1)
    data_mo['gspc'] = monthly_prices['gspc']
    print("Writing to Excel: data_monthly.xlsx")
    print(data_mo.columns)
    with pd.ExcelWriter('returns_monthly.xlsx') as writer:
        data_mo.to_excel(writer, sheet_name='monthly')
    monthly_daily = data_mo.drop('gspc', axis=1).set_index('date').resample('D').ffill()
    print(monthly_daily.tail(3))
    daily_df_all = daily_df.join([monthly_daily], how='left')
    with pd.ExcelWriter('daily.xlsx') as writer:
        daily_df_all.to_excel(writer, sheet_name='daily')
    macro_leads = create_leads(data, monthly_cols + quarterly_cols, max_lead=6)
    data_with_leads = pd.concat([data, macro_leads], axis=1)
    # Drop NaNs from the start
    corr_data = data_with_leads.dropna()

    # Example: correlations of macro leads with S&P 500 returns
    sp500_corrs = corr_data.corr()[['gspc_ret']].sort_values(by='gspc_ret', ascending=False)
    print(sp500_corrs)
