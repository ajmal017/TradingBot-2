from yahoo_fin import stock_info as si
import numpy as np
import pandas as pd
from time import sleep, time
import matplotlib.pyplot as plt

# TODO

class Trader:
    def __init__(self, stock, cash=1000, commission=10, shares=0, transaction_delay=5):
        self.stock = stock
        self.cash = cash
        self.commission = commission
        self.shares = shares
        self.cash_history = [self.cash]
        self.transaction_delay = transaction_delay
        self.update()

    def update(self):
        self.quote_table = si.get_quote_table(self.stock)

    @property
    def price(self):
        return self.quote_table["Quote Price"]

    @property
    def open_price(self):
        return self.quote_table["Open"]

    @property
    def volume(self):
        return self.quote_table["Volume"]

    @property
    def avg_volume(self):
        return self.quote_table["Avg. Volume"]

    @property
    def bid(self):
        _bid =  self.quote_table["Bid"]
        _bid = [int(_bid.split()[-1]), float(_bid.split()[0])]
        return _bid

    @property
    def ask(self):
        _ask =  self.quote_table["Ask"]
        _ask = [int(_ask.split()[-1]), float(_ask.split()[0])]
        return _ask

    @property
    def purchase_power(self):
        return int(np.floor((self.cash - 2*self.commission)/self.price))


    # Transction functions
    def pay_commission(self):
        self.cash -= self.commission

    def print_transaction_report(self):
        print("cash: " + str(np.round(self.cash, 2)))
        print("shares: " + str(self.shares))
        print("----------------------------")


    def buy(self, print_report=True):
        self.update()
        if self.shares == 0:
            self.shares += self.purchase_power
            self.cash -= self.shares*self.price
            self.pay_commission()
            if print_report:
                self.print_transaction_report()
            sleep(self.transaction_delay)

    def sell(self, print_report=True, return_reward=False):
        self.update()
        if self.shares != 0 :
            self.cash += self.shares*self.price
            self.shares -= self.shares
            self.pay_commission()
            self.cash_history.append(self.cash)
            sleep(self.transaction_delay)
            if print_report:
                self.print_transaction_report()
            if return_reward:
                return self.cash_history[-1] - self.cash_history[-2]



    def plot_progress(self):
        plt.plot(range(len(self.cash_history)), self.cash_history, "ko-")
        plt.xlabel("Trade")
        plt.ylabel("Account Balance ($)")
        plt.show()


    def get_params(self):
        self.update()
        inputs = [time(), self.price]
        return np.array(inputs)


def make_dataset(N, file_name, sleep_time):
    t = Trader("MA")
    # print(t.get_params())
    data = t.get_params()
    for _ in range(N):
        try:
            data = np.vstack((data, t.get_params()))
            sleep(sleep_time)
            print(data.shape[0])
        except Exception:
            pass
        np.save(file_name, data)

    print(data.shape)


def plot_dataset(date):
    data = np.load("data/" + date)
    plt.plot(data[:,0], data[:,1])
    # plt.plot(range(data.shape[0]), data[:,5])
    # plt.plot(range(data.shape[0]), data[:,7])
    # dpdt = (data[500:,0]-data[:-500,0])
    # plt.twinx()
    # plt.plot(range(data.shape[0]-500), dpdt, "r")

    # plt.twinx()
    # plt.plot(range(data.shape[0]-1), data[1:,2]-data[:-1,2], "r-")

    plt.show()


make_dataset(12000, "data/04_02", 2)
