import pandas as pd


data_file = 'data_final.csv'
result = pd.read_csv(data_file, header=None)
result = result.fillna(0)
print(result.head(5))
result.to_csv('data_final.csv', header=False, index=False)
print('done!')