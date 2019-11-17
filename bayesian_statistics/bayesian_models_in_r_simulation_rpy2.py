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

from numpy import inf
from scipy.stats import expon
from numpy.random import beta, standard_normal
from scipy.stats import binom, norm, lognorm
from scipy.stats import pareto
from matplotlib import colors as mcolors

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statistics

import os
os.environ['R_HOME'] = '/anaconda3/lib/R'

import rpy2.robjects as robjects


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
# np.random.seed(seed=110)
# randomSample = norm.rvs(size=size, loc=trueMu, scale=trueSig)
data = robjects.r("""
trueMu <- 5
trueSig <- 2
set.seed(100)
randomSample <- rnorm(100, trueMu, trueSig)
""")
randomSample = np.array(data)
print(['max:', max(randomSample), 'min:', min(randomSample)])

np.random.seed(seed=100)
randomSample02 = norm.rvs(size=size, loc=trueMu, scale=trueSig)

# assert any(randomSample == randomSample02)

mean = statistics.mean(randomSample)
std = statistics.stdev(randomSample, xbar=None)

print(['mean:', mean, 'std:', std])

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


'''
# Compute likelihood
lik <- sapply(1:nrow(grid), function(x){
sum(dnorm(x = randomSample, mean = grid$mu[x],
sd = grid$sigma[x], log = T))
})
'''


def function(x):
    '''
    The location (loc) keyword specifies the mean.
    The scale (scale) keyword specifies the standard deviation.

    x : array_like
            quantiles
    '''
    npdf = norm.pdf(randomSample, loc=x[0], scale=x[1])

#     npdf[npdf == -inf] = 1
#     npdf[npdf == inf] = 1

    npdf = np.log(npdf)

    return sum(npdf)


lik = map(function, grid)

lik_l = list(lik)
print([len(lik_l), lik_l])

print([max(lik_l), min(lik_l)])

'''
# Multiply (sum logs) likelihood and priors
prod <- lik + dnorm(grid$mu, mean = 0, sd = 5, log = T) +
dexp(grid$sigma, 1, log = T)
'''
# print(np.asfarray(lik_l))

prod = np.asfarray(lik_l) + np.log(norm.pdf(grid[:, 0], loc=0, scale=5)) + np.log((expon.pdf(grid[:, 1], 1)))

print([len(prod), prod])
print([max(prod), min(prod)])

'''
# Standardize the lik x prior products to sum up to 1, recover unit
prob <- exp(prod - max(prod))
'''

prob = np.exp(prod - max(prod))

# print(prob)
print(['sum(prob): ', sum(prob), 'len(prob): ', len(prob)])
print(['max(prod): ', max(prod), 'min(prod): ', min(prod)])

'''
# Sample from posterior dist of mu and sigma, plot
postSample <- sample(1:nrow(grid), size = 1e3, prob = prob)
plot(grid$mu[postSample], grid$sigma[postSample],
xlab = "Mu", ylab = "Sigma", pch = 16, col = rgb(0,0,0,.2))
abline(v = trueMu, h = trueSig, col = "red", lty = 2)
'''

'''
TypeError: 'float' object cannot be interpreted as an integer

>>> 10/5
2.0
>>> 10//5
2
>>> type(1e5)
<class 'float'>
>>> type(100000)
<class 'int'>
'''
# postSample = np.random.choice(list(range(len(grid))), size=1e3, replace=True) # TypeError: 'float' object cannot be interpreted as an integer
# postSample = np.random.choice(list(range(len(grid))), size=1000, replace=True, p=prob / prob.sum())

data = robjects.r("""
trueMu <- 5
trueSig <- 2
set.seed(100)
randomSample <- rnorm(100, trueMu, trueSig)

# Grid approximation, mu %in% [0, 10] and sigma %in% [1, 3]
grid <- expand.grid(mu = seq(0, 10, length.out = 200),
sigma = seq(1, 3, length.out = 200))

# Compute likelihood
lik <- sapply(1:nrow(grid), function(x){
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

""")
postSample = np.array(data)

# print(postSample)
print(['max(postSample): ', max(postSample), 'min(postSample): ', min(postSample)])

plt.title("Bayesian Statistics")

grid_mu = grid[:, 0][postSample]
grid_sigma = grid[:, 1][postSample]

plt.plot(grid[:, 0][postSample], grid[:, 1][postSample], 'or', color=((0, 0, 0, .2)))  # line-width (lw)
plt.xlabel('mu')
plt.ylabel('sigma')

plt.hlines(trueSig, min(grid_mu), max(grid_mu), colors='r', linestyles='dashed')
plt.vlines(trueMu, min(grid_sigma), max(grid_sigma), colors='r', linestyles='dashed')

# plt.plot(rangeP, prior / 15, 'r-')
plt.show()
