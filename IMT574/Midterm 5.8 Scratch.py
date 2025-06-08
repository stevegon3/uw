df_derm = pd.read_csv('dermatology.csv', sep='\t')
df_filtered = df_derm[df_derm['Age'] > 0]
print(df_filtered.columns)
display(df_filtered.head(3))