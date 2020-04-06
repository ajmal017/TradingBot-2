import numpy as np 
import statistics as st
import matplotlib.pyplot as plt
import datetime
import pytz
import holidays
import os

from yahoo_fin import stock_info
import time


def market_is_open():
	""" Assess if market is open """

	us_holidays = holidays.US()
	now = datetime.datetime.now()
	open_time = datetime.time(hour=9, minute=30, second=0)
	close_time = datetime.time(hour = 16, minute = 0, second = 0)

	if now.strftime('%Y-%m-%d') in us_holidays:
		return False

	if (now.time() < open_time) or (now.time() > close_time):
		return False

	if now.date().weekday() > 4:
		return False

	return True


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


class Buyer():
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


class Seller():
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

class Trader():

	def __init__(self, buyer, seller, goal, standby_time=20):
		self.buyer = buyer
		self.seller = seller
		self.data = None
		self.goal = goal
		self.standby_time = standby_time


	def test_trade(self, date, print_progress=False):
		data = np.load("data/"+date+".npy")
		data[:,0] -= data[0,0]
		self.data = data

		self.trade_prices = np.zeros((1,2))
		self.trade_times = np.zeros((1,2))
		trading = False

		#standby = [when_set, for_how_long]
		self.standby = 0

		for i in range(1, data[:,0].shape[0]):
			current_data = data[:i, :2]

			# if is not in standby
			if current_data[-1,0] > self.standby:

				if not trading:
					if self.buyer.buy(current_data):
						trading = True
						buy_time = current_data[-1,0]
						buy_price = current_data[-1,1]
						self.seller.set_brackets(buy_price)

				elif trading:
					if self.seller.sell(current_data) or \
					i == data[:,0].shape[0]-1:

						trading = False
						self.trade_prices = np.vstack((self.trade_prices, np.array([buy_price, current_data[-1,1]])))
						self.trade_times = np.vstack((self.trade_times, np.array([buy_time, current_data[-1,0]])))
						if self.buyer.buy(current_data):
							self.standby = current_data[-1,0] + self.standby_time*60


				if print_progress:
					print(np.round(i/data[:,0].shape[0]*100, 2))

				self.profits = np.round(np.sum((self.trade_prices[1:,-1] - self.trade_prices[1:,0])/self.trade_prices[1:,0]*100), 2)
		
		return self.profits


	def plot_trades(self):
		if self.data is not None:
			bollinger1, bollinger2 = bollinger(self.data[:,0], 
										self.data[:,1],
										self.buyer.ma_type,
										self.buyer.num_minutes,
										self.buyer.num_sigma)

			plt.plot(self.data[:,0]/60, self.data[:,1], "k-", label="Stock")
			plt.plot(self.data[:,0]/60, bollinger1, "b-")
			plt.plot(self.data[:,0]/60, bollinger2, "b-")
			plt.scatter(self.trade_times[1:,0]/60, self.trade_prices[1:,0], s=100, c="g", marker="o", label="Buy")
			plt.scatter(self.trade_times[1:,1]/60, self.trade_prices[1:,1], s=100, c="r", marker="x", label="Sell")
			plt.xlabel("Time (min)")
			plt.ylabel("Price ($)")
			plt.legend()
			plt.show()


	def reach_goal(self):
		profit = 0
		target = self.goal
		for time, price in zip(self.trade_times[1:,:], self.trade_prices[1:,:]):
			this_profit = (price[-1]-price[0])/price[0]*100
			profit += this_profit

			if profit > target:
				print("TradingBot achieved "+str(np.round(profit,2))+ \
					" % profit after "+str(np.round(time[-1]/60,2))+ \
					" minutes of trading.")
				return

		print("TradingBot achieved "+str(np.round(profit,2))+ \
					" % profit after a whole day of trading.")

	def max_profit(self):
		""" max profit reached throughout the day """
		if len(self.trade_prices) > 1:
			profit = 100*(self.trade_prices[1:,1]-self.trade_prices[1:,0])/self.trade_prices[1:,0]
			return np.cumsum(profit).max()
		else:
			return None



	def live_trade(self, date):

		plt.ion()
		fig = plt.figure(figsize=(20,10))
		ax = fig.add_subplot(1,1,1)

		while market_is_open():
			try:
				try:	
					profit = self.test_trade(date)
				except Exception:
					pass
				ax.clear()

				bollinger1, bollinger2 = bollinger(self.data[:,0], 
											self.data[:,1],
											self.buyer.ma_type,
											self.buyer.num_minutes,
											self.buyer.num_sigma)

				ax.plot(self.data[:,0]/60, self.data[:,1], "k-", label="Stock")
				ax.plot(self.data[:,0]/60, bollinger1, "b-")
				ax.plot(self.data[:,0]/60, bollinger2, "b-")
				ax.scatter(self.trade_times[1:,0]/60, self.trade_prices[1:,0], s=100, c="g", marker="o", label="Buy")
				ax.scatter(self.trade_times[1:,1]/60, self.trade_prices[1:,1], s=100, c="r", marker="x", label="Sell")
				ax.set_xlabel("Time (min)")
				ax.set_ylabel("Price ($)")
				plt.legend()
				plt.title(str(datetime.datetime.now()) + "  -  Total Profit: " + str(self.profits) + " %")
				plt.show()
				plt.pause(.1)

			except KeyboardInterrupt:
				break


	def test_trader(self, plot=True):
		datasets = os.listdir("data/")
		max_profits = []
		for dataset in datasets:
			if dataset[-3:] == "npy":
				profit = self.test_trade(dataset[:-4])
				if self.max_profit() is not None:	
					this_max = np.round(self.max_profit(),2)
					max_profits.append(this_max)

		if plot:
			plt.hist(max_profits, range=[-2,5], bins=28, density=True)
			plt.xlabel("Max Profit (%)")
			plt.ylabel("Occurances")
			plt.show()

		return max_profits, \
				np.round(np.mean(np.asarray(max_profits)),2), \
				np.round(st.stdev(np.asarray(max_profits)),2)





def make_dataset(sleep_time):
	data = [time.time(), stock_info.get_live_price("MA")]
	while True:
		if market_is_open():
			try:
				this_data = [time.time(), stock_info.get_live_price("MA")]
				data = np.vstack((data, this_data))
				time.sleep(sleep_time)
				print(data.shape[0])
			except Exception:
				pass
			np.save("data/"+str(datetime.datetime.now().month)+"_"+str(datetime.datetime.now().day), data)		
		else:
			print(str(datetime.datetime.now().hour) + ":" +\
				str(datetime.datetime.now().minute) + \
				" - Market is closed")
			time.sleep(3*60)
		


def mvp_trader():
	""" best trader so far """

	buyer = Buyer(ma_type="sma", num_minutes=10, num_sigma=2)
	seller = Seller(max_loss=.5, max_win=5, brackets=.5, hyst=.2)
	trader = Trader(buyer, seller, 1.5)

	return trader




