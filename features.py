import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression


def trendline(data):
    size = data.shape[0]
    y = data.reshape(-1, 1)
    x = np.array(range(size))
    x = x.reshape(-1, 1)
    mdl = LinearRegression().fit(x, y)
    return mdl.coef_[0]

def roc(a, b):
    if a != 0:
        return ((a-b)/a)
    else:
        return ((a-b)/0.01)

def percent_K(data):
    size = data.shape[0]
    high = max(data)
    low = min(data)
    if (high-low) != 0:
        return (data[size-1] - low)/(high - low)
    else:
        return (data[size-1] - low)/0.01
    
def posneg(x):
    if x > 0:
        return 1
    else:
        return 0


def calculate_features(data):
    size = data.shape[0]
    #roc_range = np.array([1, 2, 3, 5, 8, 15, 17])
    #trend_range = np.array([2, 3, 5, 8, 15, 17])
    
    feature_array = np.empty((0, 27))

    
    #num_trend_rage = trend_range.shape[0]
    #max_index = num_trend_rage - 1
    count = 19
    start = 0
    while(count <= size):
        mini_batch = data[start:count]
        mini_array = []
        mini_array1 = []
        mini_array2 = []
        mini_array3 = []
        
        f1 = [data[(count - 1), 0]]
        f2 = posneg(trendline(mini_batch[(18-2):, 1]))
        f3 = posneg(trendline(mini_batch[(18-3):, 1]))
        f4 = posneg(trendline(mini_batch[(18-5):, 1]))
        f5 = posneg(trendline(mini_batch[(18-8):, 1]))
        f6 = posneg(trendline(mini_batch[(18-15):, 1]))
        f7 = posneg(trendline(mini_batch[(18-17):, 1]))
        
        f8 = posneg(trendline(mini_batch[(17-2):18, 1]))
        f9 = posneg(trendline(mini_batch[(17-3):18, 1]))
        f10 = posneg(trendline(mini_batch[(17-5):18, 1]))
        f11 = posneg(trendline(mini_batch[(17-8):18, 1]))
        f12 = posneg(trendline(mini_batch[(17-15):18, 1]))
        f13 = posneg(trendline(mini_batch[(17-17):18, 1]))
        
        f14 = posneg(roc(mini_batch[18, 1], mini_batch[17, 1]))
        f15 = posneg(roc(mini_batch[18, 1], mini_batch[16, 1]))
        f16 = posneg(roc(mini_batch[18, 1], mini_batch[15, 1]))
        f17 = posneg(roc(mini_batch[18, 1], mini_batch[13, 1]))
        f18 = posneg(roc(mini_batch[18, 1], mini_batch[3, 1]))
        f19 = posneg(roc(mini_batch[18, 1], mini_batch[1, 1]))
        f20 = posneg(roc(mini_batch[18, 1], mini_batch[0, 1]))
        
        if f15 != 0:
            f21 = f14/f15
        else:
            f21 = f14/0.01
            
        if f16 != 0:
            f22 = f14/f16
        else:
            f22 = f14/0.01

        if f17 != 0:
            f23 = f14/f17
        else:
            f23 = f14/0.01
            
        if f18 != 0:
            f24 = f14/f18
        else:
            f24 = f14/0.0001
            
        if f19 != 0:
            f25 = f14/f19
        else:
            f25 = f14/0.01

        if f20 != 0:
            f26 = f14/f20
        else:
            f26 = f14/0.01                        
                  
        f27 = percent_K(mini_batch[(18-14):, 1])
            
       
        mini_array1 = np.append(mini_array1, [f1, f2, f3, f4, f5, f6, f7, f8, f9, f10])
        mini_array2 = np.append(mini_array2, [f11, f12, f13])
        mini_array3 = np.append(mini_array3, [f14, f15, f16, f17, f18, f19, f20, posneg(f21), posneg(f22), posneg(f23), posneg(f24), posneg(f25), posneg(f26), f27])
        
        mini_array = np.append(mini_array, mini_array1)
        mini_array = np.append(mini_array, mini_array2)
        mini_array = np.append(mini_array, mini_array3)
        
        feature_array = np.vstack((feature_array, mini_array))
        
        if count % 100 == 0:
            print(str(count) + ' of ' + str(size) + ' complete...')
        
        start = start + 1
        count = count + 1
         
    return feature_array

gold_price = pd.read_csv('gold_price.csv')
print('Calculating gold features...')
temp = pd.DataFrame(calculate_features(gold_price.as_matrix()))
temp.to_csv('gold_features.csv', header=False, index=False)
print('Gold features done!')

#---------------------------------------------------

dollar_index = pd.read_csv('dollar_index.csv')
print('Calculating dollar index features...')
temp = pd.DataFrame(calculate_features(dollar_index.as_matrix()))
temp.to_csv('dollar_index_features.csv', header=False, index=False)
print('Dollar index features done!')

#---------------------------------------------------

eur_usd = pd.read_csv('EUR_USD.csv')
print('Calculating EUR-USD features...')
temp = pd.DataFrame(calculate_features(eur_usd.as_matrix()))
temp.to_csv('EUR_USD_features.csv', header=False, index=False)
print('EUR-USD features done!')

#---------------------------------------------------

usd_gbp = pd.read_csv('USD_GBP.csv')
print('Calculating USD-GBP features...')
temp = pd.DataFrame(calculate_features(usd_gbp.as_matrix()))
temp.to_csv('USD_GBP_features.csv', header=False, index=False)
print('USD-GBP features done!')

#---------------------------------------------------

usd_cny = pd.read_csv('USD_CNY.csv')
print('Calculating USD-CNY features...')
temp = pd.DataFrame(calculate_features(usd_cny.as_matrix()))
temp.to_csv('USD_CNY_features.csv', header=False, index=False)
print('USD-CNY features done!')

#---------------------------------------------------

crude_oil = pd.read_csv('crude_oil_futures.csv')
print('Calculating crude oil features...')
temp = pd.DataFrame(calculate_features(crude_oil.as_matrix()))
temp.to_csv('crude_oil_features.csv', header=False, index=False)
print('Crude oil features done!')

#---------------------------------------------------

gold_futures = pd.read_csv('gold_futures.csv')
gold_futures['High_Low'] = gold_futures['High'] - gold_futures['Low']

print('Calculating gold futures price features...')
temp = gold_futures[['Date', 'Last']]
temp2 = pd.DataFrame(calculate_features(temp.as_matrix()))
temp2.to_csv('gold_futures_price_features.csv', header=False, index=False)
print('Gold futures price features done!')

print('Calculating gold futures volume features...')
temp = gold_futures[['Date', 'Volume']]
temp2 = pd.DataFrame(calculate_features(temp.as_matrix()))
temp2.to_csv('gold_futures_volume_features.csv', header=False, index=False)
print('Gold futures volume features done!')

print('Calculating gold futures high-low features...')
temp = gold_futures[['Date', 'High_Low']]
temp2 = pd.DataFrame(calculate_features(temp.as_matrix()))
temp2.to_csv('gold_futures_highlow_features.csv', header=False, index=False)
print('Gold futures high-low features done!')

#---------------------------------------------------

silver_futures = pd.read_csv('silver_futures.csv')
silver_futures = silver_futures.fillna(method='ffill')
silver_futures['High_Low'] = silver_futures['High'] - silver_futures['Low']

print('Calculating silver futures price features...')
temp = silver_futures[['Date', 'Last']]
temp2 = pd.DataFrame(calculate_features(temp.as_matrix()))
temp2.to_csv('silver_futures_price_features.csv', header=False, index=False)
print('Silver futures price features done!')

print('Calculating silver futures volume features...')
temp = silver_futures[['Date', 'Volume']]
temp2 = pd.DataFrame(calculate_features(temp.as_matrix()))
temp2.to_csv('silver_futures_volume_features.csv', header=False, index=False)
print('Silver futures volume features done!')

print('Calculating silver futures high-low features...')
temp = silver_futures[['Date', 'High_Low']]
temp2 = pd.DataFrame(calculate_features(temp.as_matrix()))
temp2.to_csv('silver_futures_highlow_features.csv', header=False, index=False)
print('Silver futures high-low features done!')

#---------------------------------------------------

copper_futures = pd.read_csv('copper_futures.csv')
#print(copper_futures.head())
#copper_futures = copper_futures.fillna(method='ffill')
copper_futures['High_Low'] = copper_futures['High'] - copper_futures['Low']

print('Calculating copper futures price features...')
temp = copper_futures[['Date', 'Settle']]
temp2 = pd.DataFrame(calculate_features(temp.as_matrix()))
temp2.to_csv('copper_futures_price_features.csv', header=False, index=False)
print('Copper futures price features done!')

print('Calculating copper futures volume features...')
temp = copper_futures[['Date', 'Volume']]
temp2 = pd.DataFrame(calculate_features(temp.as_matrix()))
temp2.to_csv('copper_futures_volume_features.csv', header=False, index=False)
print('Copper futures volume features done!')

print('Calculating copper futures high-low features...')
temp = copper_futures[['Date', 'High_Low']]
temp2 = pd.DataFrame(calculate_features(temp.as_matrix()))
temp2.to_csv('Copper_futures_highlow_features.csv', header=False, index=False)
print('copper futures high-low features done!')

#----------------------------------------------------------

snp_tsx = pd.read_csv('SNP_TSX.csv')
snp_tsx['High_Low'] = snp_tsx['High'] - snp_tsx['Low']

print('Calculating SNP-TSX price features...')
temp = snp_tsx[['Date', 'Close']]
temp2 = pd.DataFrame(calculate_features(temp.as_matrix()))
temp2.to_csv('snp_tsx_price_features.csv', header=False, index=False)
print('SNP-TSX price features done!')

print('Calculating SNP-TSX futures volume features...')
temp = snp_tsx[['Date', 'Volume']]
temp2 = pd.DataFrame(calculate_features(temp.as_matrix()))
temp2.to_csv('snp_tsx_volume_features.csv', header=False, index=False)
print('SNP-TSX volume features done!')

print('Calculating SNP-TSX futures high-low features...')
temp = snp_tsx[['Date', 'High_Low']]
temp2 = pd.DataFrame(calculate_features(temp.as_matrix()))
temp2.to_csv('snp_tsx_highlow_features.csv', header=False, index=False)
print('SNP-TSX  high-low features done!')

#-----------------------------------------------------------

hang_seng = pd.read_csv('hang_seng.csv')
hang_seng['High_Low'] = hang_seng['High'] - hang_seng['Low']

print('Calculating Hang Seng price features...')
temp = hang_seng[['Date', 'Close']]
temp2 = pd.DataFrame(calculate_features(temp.as_matrix()))
temp2.to_csv('hang_seng_price_features.csv', header=False, index=False)
print('Hang Seng price features done!')

print('Calculating Hang Seng volume features...')
temp = hang_seng[['Date', 'Volume']]
temp2 = pd.DataFrame(calculate_features(temp.as_matrix()))
temp2.to_csv('hang_seng_volume_features.csv', header=False, index=False)
print('Hang Seng volume features done!')

print('Calculating Hang Seng high-low features...')
temp = hang_seng[['Date', 'High_Low']]
temp2 = pd.DataFrame(calculate_features(temp.as_matrix()))
temp2.to_csv('hang_seng_highlow_features.csv', header=False, index=False)
print('Hang Seng high-low features done!')

#----------------------------------------------------------------

nikkei_225 = pd.read_csv('nikkei_225.csv')
nikkei_225 = nikkei_225.fillna(method='ffill')
nikkei_225['High_Low'] = nikkei_225['High'] - nikkei_225['Low']

print('Calculating Nikkei 225 price features...')
temp = nikkei_225[['Date', 'Close']]
temp2 = pd.DataFrame(calculate_features(temp.as_matrix()))
temp2.to_csv('nikkei_225_price_features.csv', header=False, index=False)
print('Nikkei 225 price features done!')

print('Calculating Nikkei 225 volume features...')
temp = nikkei_225[['Date', 'Volume']]
temp2 = pd.DataFrame(calculate_features(temp.as_matrix()))
temp2.to_csv('nikkei_225_volume_features.csv', header=False, index=False)
print('Nikkei 225 volume features done!')

print('Calculating Nikkei 225 high-low features...')
temp = nikkei_225[['Date', 'High_Low']]
temp2 = pd.DataFrame(calculate_features(temp.as_matrix()))
temp2.to_csv('nikkei_225_highlow_features.csv', header=False, index=False)
print('Nikkei 225 high-low features done!')



