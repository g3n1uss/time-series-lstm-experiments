"""
LSTM for international airline passengers problem with regression framing

Mostly from here http://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/
"""

import numpy
import matplotlib.pyplot as plt
from pandas import read_csv
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error


# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return numpy.array(dataX), numpy.array(dataY)

# scale or not
scale = 0  # No scaler gives an error

# fix random seed for reproducibility
numpy.random.seed(7)
# load the dataset
dataframe = read_csv('international-airline-passengers.csv', usecols=[1], engine='python', skipfooter=3)
dataset = dataframe.values
dataset = dataset.astype('float32')
max_val, max_index = max(dataset), numpy.argmax(dataset)

# define a scaler
def trans(x, al, be):
    return 1 / (1 + 1 / al * numpy.exp(be * x))

def inverse_trans(x, al, be):
    return numpy.log(al*(1-x)/x)/be

if scale==0:
    # they say LSTM works better with normalized data
    # normalize the dataset
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)
    scaled_max_val = dataset[max_index]
    # IF WE WORK WITH NORMALIZED DATA, THE LARGEST VALUE PREDICTED BY OUR NN IS LIMITED BY THE LAGEST VELUE IN THE DATASET,
    # IN THIS CASE BY 622
elif scale==1:
    min_value = min(dataset)
    print("Before normalization min is %.2f, max is %.2f" %(min(dataset),max(dataset)))
    al = 25.8967
    beta = 0.0130821
    dataset = 1/(1+1/al*numpy.exp(beta*dataset))
    print("After normalization min is %.2f, max is %.2f" % (min(dataset), max(dataset)))

# tmp = inverse_trans(trans(dataset, al, beta), al, beta)


# split into train and test sets
print('Data set has %d elements' % len(dataset))
train_size = int(len(dataset) * 0.67)
test_size = len(dataset) - train_size
# THIS SPLIT IS NOT FAIR, IT SHOULD BE DONE RANDOMLY
# BAD RESULTS FOR LARGER TIMES ARE OBTAINED BECAUSE THE MODEL IS NOT TRAINED ON LARGE VALUES
train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]

# reshape into X=t and Y=t+1
# prediction is based only on the previous state (a Markov model)
look_back = 1
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)
# reshape input to be [samples, time steps, features]
# WHY 1 IN THE MIDDLE? DOCUMENTATION SAYS IT CAN BE SKIPPED
time_steps = 1
trainX = numpy.reshape(trainX, (trainX.shape[0], time_steps, trainX.shape[1]))
testX = numpy.reshape(testX, (testX.shape[0], time_steps, testX.shape[1]))
# create and fit the LSTM network
model = Sequential()
model.add(LSTM(1, input_shape=(1, look_back)))  # number of hidden units is completely arbitrary
'''
# try stacking
model.add(LSTM(32, return_sequences=True, input_shape=(time_steps, look_back)))
model.add(LSTM(128))
'''
model.add(Dense(1))  # output is one number
# (sigmoid results in much worse predictions)
# (with relu it does not learn at all)
num_epochs = 50 # for sgd - 200, for adam - 30
model.compile(loss='mean_squared_error', optimizer='adam')
# hist = model.fit(trainX, trainY, epochs=num_epochs, batch_size=1, verbose=2)

# ==================================================
# plot learning curve
X, Y = create_dataset(dataset, look_back)
X = numpy.reshape(X, (X.shape[0], time_steps, X.shape[1]))
history = model.fit(X, Y, validation_split=0.33, epochs=num_epochs, batch_size=1, verbose=2)
# list all data in history
print(history.history.keys())
# summarize history for loss
plt.figure(1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
# plt.show()
# ===================================================

# make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

if scale==0:
    # invert predictions
    trainPredict = scaler.inverse_transform(trainPredict)
    trainY = scaler.inverse_transform([trainY])
    testPredict = scaler.inverse_transform(testPredict)
    testY = scaler.inverse_transform([testY])
elif scale==1:
    trainPredict = inverse_trans(trainPredict, al, beta)
    trainY = [inverse_trans(trainY.T, al, beta)]
    testPredict = inverse_trans(testPredict, al ,beta)
    testY = [inverse_trans(testY.T, al, beta)]

# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:, 0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:, 0]))
print('Test Score: %.2f RMSE' % (testScore))
# shift train predictions for plotting
trainPredictPlot = numpy.empty_like(dataset)
trainPredictPlot[:, :] = numpy.nan
trainPredictPlot[look_back:len(trainPredict) + look_back, :] = trainPredict
# shift test predictions for plotting
testPredictPlot = numpy.empty_like(dataset)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(trainPredict) + (look_back * 2) + 1:len(dataset) - 1, :] = testPredict
# plot baseline and predictions
plt.figure(2)
if scale==0:
    plt.plot(scaler.inverse_transform(dataset), label='Data')
elif scale==1:
    plt.plot(inverse_trans(dataset, al, beta), label='Data')
plt.plot(trainPredictPlot, label='Predictions on training data')
plt.plot(testPredictPlot, label='Predictions on testing data')
plt.legend()
plt.show()
