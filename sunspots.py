### Predicting Sunspot numbers using a Long Short-Term Memory (LSTM) Model with Keras/Tensorflow

### Data from http://www.sidc.be/silso/infosnmtot
### Useful references at https://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/

import pandas as pd
import numpy as np
import matplotlib.pyplot as pl
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, LSTM

### Sunspot data taken monthly
sunspots_monthly = pd.read_csv('SN_m_tot_V2.0.csv',sep=';',header=None)
sunspots_monthly.columns = ["year", "month", "fractionyear", "monthlymean", "monthlystd", "nobs", "definitive"]

x = sunspots_monthly["fractionyear"].values
y = sunspots_monthly["monthlymean"].values

### Normalize data
scaler = MinMaxScaler()
yscaled = scaler.fit_transform(y.reshape(-1,1))

pl.plot(x,yscaled, 'ro', alpha=0.4,ms=4)
pl.xlabel("Year")
pl.ylabel("Number")
pl.show()

### Split data into training and test sets
trainingfraction = 0.67
train_size = int(len(yscaled) * trainingfraction)
test_size = len(yscaled)-train_size
train, test = yscaled[0:train_size,:], yscaled[train_size:,:]

### The LSTM network expects input data (X) to have the array structure of [samples, time steps, features]
### Currently the data has the form [samples, features] and the problem is framed as one time step for each sample


### Function to create a new dataset
def create_dataset(dataset, look_back):
	### INPUT:	dataset = Numpy array
	### 		look_back = number of previous time steps to use as input variables to predict the next time period
	X, Y = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), 0]
		X.append(a)
		Y.append(dataset[i + look_back, 0])
	return np.array(X), np.array(Y)

### Reshape data into X = t and Y = t+1
look_back = 1
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

### Now reshape input to be [samples, time steps, features]
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))


### Create and fit the LSTM network
### The network has a visible layer with 1 input, a hidden layer with 4 LSTM blocks or neurons
### and an output layer that makes a single value prediction. The default sigmoid activation 
### function is used for the LSTM blocks. The network is trained for 10 epochs and a batch 
### size of 1 is used.
model = Sequential()
model.add(LSTM(4, input_shape=(1, look_back)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=10, batch_size=1, verbose=2)

### Predict the training and test data
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

### Invert the predictions back to original data scale
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])

### Calculate root mean squared error
trainScore = np.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print("Train Score: {:.2f} RMS".format(trainScore))
testScore = np.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print("Test Score: {:.2f} RMSE".format(testScore))

### Plot data and predictions
### First shift train and test predictions for plotting
trainPredictPlot = np.empty_like(yscaled)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
testPredictPlot = np.empty_like(yscaled)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(yscaled)-1, :] = testPredict

pl.plot(scaler.inverse_transform(yscaled),'ro',alpha=0.4,ms=4,label='Data')
pl.plot(trainPredictPlot, 'b-',label='Train')
pl.plot(testPredictPlot, 'g-',label='Test')
pl.legend()
pl.show()