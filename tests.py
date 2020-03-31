import numpy as np 
# from trader import *
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


class Buyer:
	def __init__(self, ma_type="sma", num_minutes=20, num_sigma=2):
		self.ma_type = ma_type
		self.num_minutes = num_minutes
		self.num_sigma = num_sigma

	def buy(self, data):
		""" Algo to get in a trade """

		time, price = data[:,0], data[:,1]

		considered = np.where(time >= time[-1] - 60*self.num_minutes)

		if time[considered][-1] - time[considered][0] < 59*self.num_minutes:
			return False

		else:
			if self.ma_type is "sma":
				ma = st.mean(price[considered])
			elif self.ma_type is "ema":
				ma = ema(price[considered])

			std = st.stdev(price[considered])
			bottom_bollinger = ma - self.num_sigma*std

			if price[-1] < bottom_bollinger:
				return True
			else:
				return  False


class Seller:
	def __init__(self, max_loss=1, max_win=5, brackets=.5, hyst=.1):
		self.max_loss = max_loss
		self.max_win = max_win
		self.brackets = brackets
		self.hyst = hyst

	def set_brackets(self, buy_price):
		# calculating price brackets
		self.bottom = buy_price - buy_price*self.max_loss/100
		self.top = buy_price + buy_price*self.max_win/100
		hysteresis = self.hyst/100*buy_price

		levels = np.arange(self.brackets, self.max_win, self.brackets)
		levels_bottom = buy_price + buy_price*levels/100
		levels_bottom = np.append(self.bottom, levels_bottom)

		levels_top = levels_bottom[1:] + hysteresis
		levels_top = np.append(levels_top, self.top)

		self.levels = np.vstack((levels_bottom, levels_top))
		self.buy_price = buy_price
		self.current_bracket = 0

	def sell(self, data):
		""" Algo to get out of a trade """

		price = data[-1,1]
		bracket = self.levels[:,self.current_bracket]

		if price >= self.top:
			return True # max profit reached

		elif price <= bracket[0]:
			return True # below lower bracket

		elif price >= bracket[1]: # above top bracket
			self.current_bracket += 1
			return False
		else: # still between brackets
			return False


def test_trade(date, buyer, seller):
	data = np.load("data/"+date+".npy")
	data[:,0] -= data[0,0]

	trade_prices = np.zeros((1,2))
	trade_times = np.zeros((1,2))
	trading = False

	for i in range(1, data[:,0].shape[0]):
		current_data = data[:i, :2]

		if not trading:
			if buyer.buy(current_data):
				trading = True
				buy_time = current_data[-1,0]
				buy_price = current_data[-1,1]
				seller.set_brackets(buy_price)

		elif trading:
			if seller.sell(current_data) or i == data[:,0].shape[0]-1:
				trading = False
				trade_prices = np.vstack((trade_prices, np.array([buy_price, current_data[-1,1]])))
				trade_times = np.vstack((trade_times, np.array([buy_time, current_data[-1,0]])))

		print(np.round(i/data[:,0].shape[0]*100, 2))

	return (trade_times[1:,:], trade_prices[1:,:])


def assess_performance(trades, plot=False, data=None):
	times, prices = trades
	profit = np.sum((prices[:,-1] - prices[:,0])/prices[:,0]*100)

	if plot:
		plt.plot(data[:,0]/60, data[:,1], "k-", label="Stock")
		plt.plot(times[:,0]/60, prices[:,0], "go", label="Buy")
		plt.plot(times[:,1]/60, prices[:,1], "ro", label="Sell")
		plt.legend()
		plt.show()

	return profit



date = "03_30"
buyer = Buyer(ma_type="sma", num_minutes=10, num_sigma=2)
seller = Seller(max_loss=.5, max_win=5, brackets=.5, hyst=.2)

trades = test_trade(date, buyer, seller)
data = np.load("data/03_30.npy")
data[:,0] -= data[0,0]
profit = assess_performance(trades, True, data)

print(np.round(profit, 2))



top, bottom = bollinger(data[:,0], data[:,1], "sma", 10, 2)
# # top_, bottom_ = bollinger(data[:,0], data[:,1], "ema", 100, 2)

plt.plot(data[:,0]/60, data[:,1], "r")
plt.plot(data[:,0]/60, top, "k")
plt.plot(data[:,0]/60, bottom, "k")
# # plt.plot(data[:,0], top_, "b")
# # plt.plot(data[:,0], bottom_, "b")
# plt.twinx()
# plt.plot(data[1:,0]/60, data[1:,3]-data[:-1,3])
plt.show()

# buying = []
# for i in range(data[:,0].shape[0]):
# 	if buy(data[:i+1, 0:2]):
# 		buying.append(1)
# 	else:
# 		buying.append(0)
# plt.plot(data[:,0]/60, buying)
# plt.twinx()
# plt.plot(data[:,0]/60, data[:,1])
# plt.show()
# def train(data, buyer, seller):



# prices = [200, 200, 199, 200.5, 201, 203, 204, 205, 206, 206.5, 206.3, 206.2, 206.1, 206, 205.9, 205.1]
# make_profit(200)



# data_rally = data[data[:,0]>1000]
# prices = data_rally[:,1]

# make_profit(prices[0], brackets=.6, hyst = .3)
# plt.figure()
# plt.plot(data[:,0], top-bottom)
# plt.show()

# import sub__process




