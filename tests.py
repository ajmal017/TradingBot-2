import numpy as np 
from trader import *
from Q_models import *
import torch
import statistics as st

# data = torch.Tensor(np.load("data/03_25"))


# qnet = Qnet_linear(data[0].shape[0], 3)
# print(qnet(data[0:3]).data.numpy())
# print(np.argmax(qnet(data[0:3]).data.numpy()))
# # for i in range(15):
# # 	print(qnet(data[i]).data.numpy())

# print(sum(p.numel() for p in qnet.parameters() if p.requires_grad))

data = np.load("data/03_30.npy")
data[:,0] -= data[0,0]
times = data[:,0]
# data = data[times < 5000, :]
# plt.plot(data[:,0], data[:,1])
# plt.show()


def ema(values):
	""" https://www.investopedia.com/terms/e/ema.asp """
	ema = values[0]
	smoothing = .1 #20/(values.shape[0] + 1)
	for i in range(1,values.shape[0]):
		ema = values[i]*smoothing + ema*(1 - smoothing)

	return ema


def bollinger(time, price, ma_type, num_minutes=20, num_sigma=2):

	top = price[0]*np.ones(price.shape[0])
	bottom = price[0]*np.ones(price.shape[0])

	for i in range(1,price.shape[0]):

		condition1 = time <= time[i]
		condition2 = time >= time[i] - 60*num_minutes
		idx = np.where(np.logical_and(condition1, condition2)) 

		if ma_type is "sma":
			ma = st.mean(price[idx])
		elif ma_type is "ema":
			ma = ema(price[idx])

		std = st.stdev(price[idx])
		top[i] = ma + num_sigma*std
		bottom[i] = ma - num_sigma*std
	
	return top, bottom

top, bottom = bollinger(data[:,0], data[:,1], "sma", 20, 2)
# top_, bottom_ = bollinger(data[:,0], data[:,1], "ema", 100, 2)

plt.plot(data[:,0]/60, data[:,1], "r")
plt.plot(data[:,0]/60, top, "k")
plt.plot(data[:,0]/60, bottom, "k")
# plt.plot(data[:,0], top_, "b")
# plt.plot(data[:,0], bottom_, "b")
plt.twinx()
plt.plot(data[1:,0]/60, data[1:,3]-data[:-1,3])
plt.show()





def make_profit(buy_price, max_loss=1, max_win=5, brackets=.5, hyst=.1):
	""" Algo to protect profits """

	# calculating prices
	bottom = buy_price - buy_price*max_loss/100
	top = buy_price + buy_price*max_win/100
	hysteresis = hyst/100*buy_price

	levels = np.arange(brackets, max_win, brackets)
	levels_bottom = buy_price + buy_price*levels/100
	levels_bottom = np.append(bottom, levels_bottom)

	levels_top = levels_bottom[1:] + hysteresis
	levels_top = np.append(levels_top, top)

	levels = np.vstack((levels_bottom, levels_top))

	trading = True
	# initial bracket
	i = 0
	j = 0 # for simulation only
	while trading:
		bracket = levels[:,i]
		price = prices[j] # fetch_price()

		if price >= top:
			action = "sell -- profit: " + str((price-buy_price)/buy_price*100) + " %" # sell()
			trading = False

		elif price <= bracket[0]:
			action = "sell -- profit: " + str((price-buy_price)/buy_price*100) + " %" # sell()
			trading = False

		elif price >= bracket[1]:
			i += 1
			action = "level-up"
		else:
			action = "hold"
		
		print(np.round(buy_price,2), np.round(price,2), np.round(bracket[0],2), np.round(bracket[1],2), action)

		j += 1

# prices = [200, 200, 199, 200.5, 201, 203, 204, 205, 206, 206.5, 206.3, 206.2, 206.1, 206, 205.9, 205.1]
# make_profit(200)



# data_rally = data[data[:,0]>1000]
# prices = data_rally[:,1]

# make_profit(prices[0], brackets=.6, hyst = .3)
# plt.figure()
# plt.plot(data[:,0], top-bottom)
# plt.show()

# import sub__process




