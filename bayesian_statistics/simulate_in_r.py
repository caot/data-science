from numpy.random import beta
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

'''
page 224 of book <<An Introduction to the Science of Statistics>>
                 https://www.math.arizona.edu/~jwatkins/statbook.pdf

> p1<-rbeta(10000,16,6);p2<-rbeta(10000,18,4)
> p<-p1*p2

We then give a table of deciles for the posterior distribution function and present a histogram.

> d<-seq(0, 1, by=0.1)
> data.frame(quantile(p,d))
'''
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('max_colwidth', -1)

np.set_printoptions(threshold=np.Infinity)

d = np.linspace(0, 1, 11)
print(['d: ', d])


def quantile():
    p1 = beta(16, 6, 10000)
    p2 = beta(18, 4, 10000)

    p = p1 * p2

    return np.quantile(p, d)


df = pd.DataFrame({
    'quantile A': quantile(),
    'quantile B': quantile(),
}, index=["{0:.0f}%".format(val * 100) for val in d])


pct_change = df.pct_change(axis='columns')
print(pct_change)


df.reset_index().plot(x='index')

plt.show()
