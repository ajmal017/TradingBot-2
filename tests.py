# import numpy as np 
# from datetime import datetime, timedelta
# from utils import *
import os
import matplotlib.pyplot as plt 
import datetime

def test_trader(self, plot=True):
	datasets = os.listdir("data/")
	max_profits = np.empty(0)
	for dataset in datasets:
		if dataset[-3:] == "npy":
			profit = self.test_trade(dataset[:-4])
			this_max = self.max_profit()
			max_profits = np.append(max_profits, this_max)

	if plot:
		plt.hist(max_profits, bins=10)
		plt.xlabel("Max Profit ($)")
		plt.ylabel("Occurances")
		plt.show()

	return max_profits



print(datetime.datetime.now().hour)