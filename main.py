from utils import *


buyer = Buyer(ma_type="sma", num_minutes=10, num_sigma=2)
seller = Seller(max_loss=.5, max_win=5, brackets=.5, hyst=.2)
trader = Trader(buyer, seller)
date = "03_30"

profit = trader.test_trade(date)
trader.two_percent()
# print(profit)
# trader.plot_trades()


