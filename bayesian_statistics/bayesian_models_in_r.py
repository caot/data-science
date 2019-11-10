'''
https://www.r-bloggers.com/bayesian-models-in-r-2/

> rangeP <- seq(0, 1, length.out = 100)
> plot(rangeP, dbinom(x = 8, prob = rangeP, size = 10), type = "l", xlab = "P(Black)", ylab = "Density")

> lines(rangeP, dnorm(x = rangeP, mean = .5, sd = .1) / 15, col = "red")

> lik <- dbinom(x = 8, prob = rangeP, size = 10)
> prior <- dnorm(x = rangeP, mean = .5, sd = .1)

> unstdPost <- lik * prior
> lines(rangeP, unstdPost, col = "green")

> stdPost <- unstdPost / sum(unstdPost)
> lines(rangeP, stdPost, col = "blue")

> legend("topleft", legend = c("Lik", "Prior", "Unstd Post", "Post"), text.col = 1:4, bty = "n")
>


Equivalence between distribution functions in R and Python

The name for the different functions that work with probability distributions in R and SciPy is different, which is often confusing. The following table lists the equivalence between the main functions:
R                           SciPy     Name
dnorm()                     pdf()     Probability density function (PDF)
pnorm()                     cdf()     Cumulative density function (CDF)
qnorm()                     ppf()     Percentile point function (CDF inverse)
pnorm(lower.tail = FALSE)   sf()      Complementary CDF (CCDF) or survival function
qnorm(lower.tail = FALSE)   isf()     CCDF inverse or inverse survival function
rnorm()                     rvs()     Random samples
'''

from numpy.random import beta
from scipy.stats import binom, norm
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

from matplotlib import colors as mcolors

colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)

d = np.linspace(0, 1, 11)

rangeP = np.linspace(0, 1, 100 + 1)

'''
The location (loc) keyword specifies the mean.
The scale (scale) keyword specifies the standard deviation.
'''
npdf = norm.pdf(rangeP, loc=0.5, scale=0.1)

'''
x = 8
size = 10,
prob=rangeP
'''
lik = binom.pmf(8, 10, rangeP)
prior = norm.pdf(rangeP, loc=0.5, scale=0.1)

plt.title("Bayesian Statistics")

plt.plot(rangeP, lik, 'k-')  # line-width (lw)
plt.xlabel('P(Black) / Probability range')
plt.ylabel('Density / pmf Probability mass function')

plt.plot(rangeP, prior / 15, 'r-')

unstdPost = lik * prior
print([max(unstdPost), min(unstdPost)])

stdPost = unstdPost / sum(unstdPost)

plt.plot(rangeP, unstdPost, color='limegreen', linestyle='solid')
plt.plot(rangeP, stdPost, 'b-')

ax = plt.gca()
plt.gca().legend(('lik', 'prior', 'unstdPost', 'stdPost'))

plt.show()
