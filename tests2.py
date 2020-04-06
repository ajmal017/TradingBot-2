from yahoo_fin import stock_info as si

# print(si.get_live_price("MA"))

import numpy as np 
import matplotlib.pyplot as plt 

v = [-2,-2,-2,-1.5,-1.2,-.75,0,1.5,2.6,4.9,5,1,2,3,4,5,-1,-2,4,2,2,2,3,4]

histo, edges = np.histogram(v, range=[-2,5], bins=10, density=True)
print(histo)
plt.hist(v, range=[-2,5], bins=28, density=True)
plt.show()