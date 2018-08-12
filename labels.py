import quandl
import pandas as pd
import numpy as np

#gold_price = quandl.get("WGC/GOLD_DAILY_USD", authtoken="tbTSHmzg4fvb5zRiCGdU", start_date="1980-01-01", end_date="2017-01-02")
#gold_price.to_csv('gold_price_labels.csv')
#print('done!')

gold_price = pd.read_csv('gold_price_labels.csv')
size = gold_price.shape[0]
count = 0
label_array = np.empty((0, 2))

#print(gold_price.iloc[1, 1])

while(count < (size-1)):
    delta = gold_price.iloc[(count + 1), 1] - gold_price.iloc[count, 1]
    
    if delta > 0:
        mini_array = np.array([gold_price.iloc[count, 0], 1])
        label_array = np.vstack((label_array, mini_array))
    else:
        mini_array = np.array([gold_price.iloc[count, 0], 0])
        label_array = np.vstack((label_array, mini_array))
    
    count += 1

labels_pd = pd.DataFrame(label_array)
print(labels_pd.head(30))
print(labels_pd.shape)
data_final = pd.read_csv('data_final.csv', header=None)
print(data_final.shape)
data_final_dates = data_final[[0]]
print(data_final_dates.shape)

labels_final = labels_pd[18:]
#pd.merge(data_final_dates, labels_pd, how='inner', on=[0])
print(labels_final.head())
print(labels_final.shape)
labels_final.to_csv('labels_final.csv', header=False, index=False)
print('done!')