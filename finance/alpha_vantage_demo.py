'''
alpha_vantage, a module for grabbing financial investments data from Alpha
Vantage APIs maintained by a company I randomly came across called Alpha Vantage
Inc, which provides free access to APIs for retrieving real-time and historical
stock, forex (FX), and digital/crypto currencies data (this one will need to be
installed using AUR and corresponding syntax, as follows)

yaourt -S python-alpha_vantage

'''

# Pulling and plotting VXUS ETF price data# Import the software libraries whose functionality we will rely on
from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.techindicators import TechIndicators

from matplotlib.pyplot import figure

import matplotlib.pyplot as plt  # Input your unique Alpha Vantage API key on the next line

key = 'INPUT YOUR KEY HERE'  # Choose your output format (pandas tabular), or default to JSON (python dict)
ts = TimeSeries(key, output_format='pandas')  # Get the data, which returns a tuple of 100 records
# vxus_data is a pandas dataframe, vxus_meta_data is a dict

figure(num=None, figsize=(15, 6), dpi=80, facecolor='w', edgecolor='k')

vxus_data, vxus_meta_data = ts.get_daily(symbol='VXUS')  # Visualize the data by determining the plot size and style

print(type(vxus_data), vxus_data)

# vxus_data = vxus_data[::-1]

vxus_data['4. close'].plot()

# plt.gca().invert_xaxis()

plt.tight_layout()
plt.grid()
plt.show()
