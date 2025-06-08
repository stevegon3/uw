import pandas as pd

columns = pd.read_csv('cnames.txt')['Name'].tolist()
df = pd.DataFrame([[0 for x in range(len(columns))]], columns = columns)
n = 0
for f in ['GL2010.TXT', 'GL2011.TXT', 'GL2012.TXT', 'GL2013.TXT']:
    data = pd.read_csv(f)
    n += len(data)
    print(n, len(data))
    df = pd.concat([df, data])
print(len(df))
if len(df) == 9720:
    print("The data does not add up to 9720 rows")
display(df)