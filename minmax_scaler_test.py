from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0, 1))

dataset = [1, 2, 3, 4]
scaler.fit(dataset)
dataset = scaler.transform(dataset)

print(scaler.transform([5]))
print(scaler.inverse_transform([0.8, 1, 2]))
