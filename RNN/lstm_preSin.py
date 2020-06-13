#https://www.youtube.com/watch?v=LiBFV7ptm4M
#http://apmonitor.com/do/index.php/Main/LSTMNetwork
import numpy as np
import matplotlib.pyplot as plt

# data generation & data preparation
 n = 500
 t = np.linspace(0,20.0*np.pi,n)
 X = np.sin(t)

 window = 10
 last = int(n/5.0)#最后100个(1/5)
 Xtrain = X[:-last]
 Xtest = X[-last-window:]

xin = []
next_X = []
for i in range(window,len(Xtrain)):
	xin.append(Xtrain[i-window:i]) #分为390组，每组10个，且每组顺移一位
	next_X.append(Xtrain[i]) #每组10个的后一个,预测值

xin, next_X = np.array(xin),np.array(next_X)#list转np数组,[390,10]
xin = xin.reshape(xin.shape[0],xin.shape[1],1) #[390,10,1]


# Model Structure
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout

m = Sequential()
m.add(LSTM(units=50, return_sequences=True,input_shape=(xin.shape[1],1))) #[-1,10,1]
m.add(Dropout(0.2))
m.add(LSTM(units=50))
m.add(Dropout(0.2))
m.add(Dense(units=1))
m.compile(optimizer='adam', loss = 'mean_squared_error')

#Training
history = m.fit(xin, next_X, epochs=50, batch_size=50, verbose=0)
plt.figure()
plt.ylabel('loss')
plt.xlabel('epoch')
plt.semilogy(history.history['loss'])
plt.show()

#Validation
xin = []
next_X1 = []
for i in range(window,len(Xtest)):
	xin.append(Xtest[i-window:i])
	next_X1.append(Xtest[i])

xin, next_X1 = np.array(xin), np.array(next_X1)
xin = xin.reshape((xin.shape[0],xin.shape[1],1))

X_pred = m.predict(xin) #(100,10,1) -> (100,1)

plt.figure()
plt.plot(X_pred,':',label='LSTM')
plt.plot(next_X1,'--',label='Actual')
plt.legend()

#Forecast
X_pred = Xtext.copy()
for i in range(window,len(X_pred)):
	xin = X_pred[i-window:i].reshape((1,window,1))
	X_pred[i] = m.predict(xin)

plt.figure()
plt.plot(X_pred[window:],':',label='LSTM')
plt.plot(next_X1,'--',label='Actual')#(100,)
plt.legend()

#Forecast2
X = Xtext.copy()
for i in range(100):
	xin = X[i:i+10]
	xin = xin.reshape((1,10,1))
	X[i+10] = m.predict(xin)

plt.figure()
plt.plot(X,':',label='LSTM')
plt.plot(Xtext,'--',label='Actual')#(100,)
plt.legend()
