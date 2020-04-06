from utils import *


trader = mvp_trader()
date = str(datetime.datetime.now().month)+"_"+str(datetime.datetime.now().day)


# Daily performance
if market_is_open():
	trader.live_trade(date)
else:
	if 0:
		profit = trader.test_trade(date)
		trader.plot_trades()
		trader.reach_goal()
		print(profit)

# Study model on all datasets		
if 0:

	trader.goal = 3
	trader.buyer.num_minutes = 10
	trader.standby_time = 5

	profits, mean, std = trader.test_trader(True)
	# print("average of "+str(mean)+" +- "+str(std))
	profits = np.asarray(profits)
	profits[profits >= trader.goal] = trader.goal
	print(np.mean(profits))