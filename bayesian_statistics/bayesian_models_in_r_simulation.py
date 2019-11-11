'''
https://www.r-bloggers.com/bayesian-models-in-r-2/

Simulation

# Define real pars mu and sigma, sample 100x
trueMu <- 5
trueSig <- 2
set.seed(100)
randomSample <- rnorm(100, trueMu, trueSig)
# Grid approximation, mu %in% [0, 10] and sigma %in% [1, 3]
grid <- expand.grid(mu = seq(0, 10, length.out = 200),
sigma = seq(1, 3, length.out = 200))
# Compute likelihood
lik <- sapply(1:nrow(grid), ffuunnccttiioonn(x){
sum(dnorm(x = randomSample, mean = grid$mu[x],
sd = grid$sigma[x], log = T))
})
# Multiply (sum logs) likelihood and priors
prod <- lik + dnorm(grid$mu, mean = 0, sd = 5, log = T) +
dexp(grid$sigma, 1, log = T)
# Standardize the lik x prior products to sum up to 1, recover unit
prob <- exp(prod - max(prod))
# Sample from posterior dist of mu and sigma, plot
postSample <- sample(1:nrow(grid), size = 1e3, prob = prob)
plot(grid$mu[postSample], grid$sigma[postSample],
xlab = "Mu", ylab = "Sigma", pch = 16, col = rgb(0,0,0,.2))
abline(v = trueMu, h = trueSig, col = "red", lty = 2)

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

from numpy.random import beta, standard_normal
from scipy.stats import binom, norm
# from scipy.stats.rv_continuous import rvs
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import statistics

from scipy.stats import pareto
from matplotlib import colors as mcolors

colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)

b = 0.9

np.random.seed(seed=200)
case01 = pareto.rvs(b, loc=0, scale=1, size=5)

np.random.seed(seed=200)
case02 = pareto.rvs(b, loc=0, scale=1, size=5)

assert any(case01 == case02)

'''
rnorm2 <- function(n,mean,sd) { mean+sd*scale(rnorm(n)) }
r <- rnorm2(100,4,1)
mean(r)  ## 4
sd(r)    ## 1
'''

'''
# Define real pars mu and sigma, sample 100x
trueMu <- 5
trueSig <- 2
set.seed(100)
randomSample <- rnorm(100, trueMu, trueSig)
'''
trueMu = 5
trueSig = 2
size = 100

# rangeP = np.linspace(4, 6, 100 + 1)
# print(rangeP)

'''
The location (loc) keyword specifies the mean.
The scale (scale) keyword specifies the standard deviation.
'''
np.random.seed(seed=100)
randomSample = norm.rvs(size=size, loc=trueMu, scale=trueSig)

np.random.seed(seed=100)
randomSample02 = norm.rvs(size=size, loc=trueMu, scale=trueSig)

assert any(randomSample == randomSample02)

mean = statistics.mean(randomSample)
std = statistics.stdev(randomSample, xbar=None)

print([mean, std])


'''
# Grid approximation, mu %in% [0, 10] and sigma %in% [1, 3]
grid <- expand.grid(mu = seq(0, 10, length.out = 200),
sigma = seq(1, 3, length.out = 200))
'''
mu = np.linspace(0, 10, 200)
# print(mu)

sigma = np.linspace(1, 3, 200)
# print(sigma)

grid = np.array(np.meshgrid(mu, sigma)).reshape(2, 200 * 200).T

print(grid)
