import pandas as pd


gold_price_features = pd.read_csv('gold_features.csv', header=None)
dollar_index_features = pd.read_csv('dollar_index_features.csv', header=None)
EUR_USD_features = pd.read_csv('EUR_USD_features.csv', header=None)
USD_GBP_features = pd.read_csv('USD_GBP_features.csv', header=None)
USD_CNY_features = pd.read_csv('USD_CNY_features.csv', header=None)
crude_oil_features = pd.read_csv('crude_oil_features.csv', header=None)

gold_futures_price_features = pd.read_csv('gold_futures_price_features.csv', header=None)
gold_futures_volume_features = pd.read_csv('gold_futures_volume_features.csv', header=None)
gold_futures_highlow_features = pd.read_csv('gold_futures_highlow_features.csv', header=None)

silver_futures_price_features = pd.read_csv('silver_futures_price_features.csv', header=None)
silver_futures_volume_features = pd.read_csv('silver_futures_volume_features.csv', header=None)
silver_futures_highlow_features = pd.read_csv('silver_futures_highlow_features.csv', header=None)

copper_futures_price_features = pd.read_csv('copper_futures_price_features.csv', header=None)
copper_futures_volume_features = pd.read_csv('copper_futures_volume_features.csv', header=None)
copper_futures_highlow_features = pd.read_csv('copper_futures_highlow_features.csv', header=None)

snp_tsx_price_features = pd.read_csv('snp_tsx_price_features.csv', header=None)
snp_tsx_volume_features = pd.read_csv('snp_tsx_volume_features.csv', header=None)
snp_tsx_highlow_features = pd.read_csv('snp_tsx_highlow_features.csv', header=None)

hang_seng_price_features = pd.read_csv('hang_seng_price_features.csv', header=None)
hang_seng_volume_features = pd.read_csv('hang_seng_volume_features.csv', header=None)
hang_seng_highlow_features = pd.read_csv('hang_seng_highlow_features.csv', header=None)

nikkei_225_price_features = pd.read_csv('nikkei_225_price_features.csv', header=None)
nikkei_225_volume_features = pd.read_csv('nikkei_225_volume_features.csv', header=None)
nikkei_225_highlow_features = pd.read_csv('nikkei_225_highlow_features.csv', header=None)

print(gold_price_features.shape)
result = pd.merge(gold_price_features, dollar_index_features, how='left', on=[0])
total_features = result.shape[1]
result.columns = range(total_features)
print(result.shape)

print(EUR_USD_features.shape)
result = pd.merge(result, EUR_USD_features, how='left', on=[0])
total_features = result.shape[1]
result.columns = range(total_features)
print(result.shape)

print(USD_GBP_features.shape)
result = pd.merge(result, USD_GBP_features, how='left', on=[0])
total_features = result.shape[1]
result.columns = range(total_features)
print(result.shape)

print(USD_CNY_features.shape)
result = pd.merge(result, USD_CNY_features, how='left', on=[0])
total_features = result.shape[1]
result.columns = range(total_features)
print(result.shape)

print(crude_oil_features.shape)
result = pd.merge(result, crude_oil_features, how='left', on=[0])
total_features = result.shape[1]
result.columns = range(total_features)
print(result.shape)

#----------------------------------------------------

print(gold_futures_price_features.shape)
result = pd.merge(result, gold_futures_price_features, how='left', on=[0])
total_features = result.shape[1]
result.columns = range(total_features)
print(result.shape)

print(gold_futures_volume_features.shape)
result = pd.merge(result, gold_futures_volume_features, how='left', on=[0])
total_features = result.shape[1]
result.columns = range(total_features)
print(result.shape)

print(gold_futures_highlow_features.shape)
result = pd.merge(result, gold_futures_highlow_features, how='left', on=[0])
total_features = result.shape[1]
result.columns = range(total_features)
print(result.shape)

#----------------------------------------------------

print(silver_futures_price_features.shape)
result = pd.merge(result, silver_futures_price_features, how='left', on=[0])
total_features = result.shape[1]
result.columns = range(total_features)
print(result.shape)

print(silver_futures_volume_features.shape)
result = pd.merge(result, silver_futures_volume_features, how='left', on=[0])
total_features = result.shape[1]
result.columns = range(total_features)
print(result.shape)

print(silver_futures_highlow_features.shape)
result = pd.merge(result, silver_futures_highlow_features, how='left', on=[0])
total_features = result.shape[1]
result.columns = range(total_features)
print(result.shape)

#----------------------------------------------------

print(copper_futures_price_features.shape)
result = pd.merge(result, copper_futures_price_features, how='left', on=[0])
total_features = result.shape[1]
result.columns = range(total_features)
print(result.shape)

print(copper_futures_volume_features.shape)
result = pd.merge(result, copper_futures_volume_features, how='left', on=[0])
total_features = result.shape[1]
result.columns = range(total_features)
print(result.shape)

print(copper_futures_highlow_features.shape)
result = pd.merge(result, copper_futures_highlow_features, how='left', on=[0])
total_features = result.shape[1]
result.columns = range(total_features)
print(result.shape)

#----------------------------------------------------

print(snp_tsx_price_features.shape)
result = pd.merge(result, snp_tsx_price_features, how='left', on=[0])
total_features = result.shape[1]
result.columns = range(total_features)
print(result.shape)

print(snp_tsx_volume_features.shape)
result = pd.merge(result, snp_tsx_volume_features, how='left', on=[0])
total_features = result.shape[1]
result.columns = range(total_features)
print(result.shape)

print(snp_tsx_highlow_features.shape)
result = pd.merge(result, snp_tsx_highlow_features, how='left', on=[0])
total_features = result.shape[1]
result.columns = range(total_features)
print(result.shape)

#----------------------------------------------------

print(hang_seng_price_features.shape)
result = pd.merge(result, hang_seng_price_features, how='left', on=[0])
total_features = result.shape[1]
result.columns = range(total_features)
print(result.shape)

print(hang_seng_volume_features.shape)
result = pd.merge(result, hang_seng_volume_features, how='left', on=[0])
total_features = result.shape[1]
result.columns = range(total_features)
print(result.shape)

print(hang_seng_highlow_features.shape)
result = pd.merge(result, hang_seng_highlow_features, how='left', on=[0])
total_features = result.shape[1]
result.columns = range(total_features)
print(result.shape)

#----------------------------------------------------

print(nikkei_225_price_features.shape)
result = pd.merge(result, nikkei_225_price_features, how='left', on=[0])
total_features = result.shape[1]
result.columns = range(total_features)
print(result.shape)

print(nikkei_225_volume_features.shape)
result = pd.merge(result, nikkei_225_volume_features, how='left', on=[0])
total_features = result.shape[1]
result.columns = range(total_features)
print(result.shape)

print(nikkei_225_highlow_features.shape)
result = pd.merge(result, nikkei_225_highlow_features, how='left', on=[0])
total_features = result.shape[1]
result.columns = range(total_features)
print(result.shape)

result = result.fillna(method='ffill')
result = result.fillna(method='bfill')
print(result.head())
print(result.tail())

result.to_csv('data_final.csv', header=False, index=False)
print('done!')

