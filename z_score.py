# -*- encoding: utf-8 -*-

from z_table import to_cumulative, to_col

'''
https://en.wikipedia.org/wiki/Standard_score

    For Fisher z-transformation in statistics, see Fisher transformation.
    For Z-values in ecology, see Z-value.
    For z-transformation to complex number domain, see Z-transform.
    For Z-factor in high-throughput screening, see Z-factor.
    For Z-score financial analysis tool, see Altman Z-score.

In statistics, the standard score is the signed fractional number of standard deviations by
which the value of an observation or data point is above the mean value of what is being
observed or measured. Observed values above the mean have positive standard scores, while
values below the mean have negative standard scores.

It is calculated by subtracting the population mean from an individual raw score and then
dividing the difference by the population standard deviation. It is a dimensionless quantity.
This conversion process is called standardizing or normalizing (however, "normalizing" can
refer to many types of ratios; see normalization for more).

Standard scores are also called z-values, z-scores, normal scores, and standardized variables.
They are most frequently used to compare an observation to a theoretical deviate, such as a
standard normal deviate.


Calculation

If the population mean and population standard deviation are known, the standard score of
a raw score x[1] is calculated as

z = (x - μ) / σ

where:

    μ is the mean of the population.
    σ is the standard deviation of the population.

The absolute value of z represents the distance between the raw score and the population
mean in units of the standard deviation. z is negative when the raw score is below the mean,
positive when above.

Calculating z using this formula requires the population mean and the population standard
deviation, not the sample mean or sample deviation. But knowing the true mean and standard
deviation of a population is often unrealistic except in cases such as standardized testing,
where the entire population is measured.

When the population mean and the population standard deviation are unknown, the standard
score may be calculated using the sample mean and sample standard deviation as estimates of
the population values.

In these cases, the z score is

    z = x − x ¯ S {\displaystyle z={x-{\bar {x}} \over S}} {\displaystyle z={x-{\bar {x}} \over S}}

where:

    x ¯ {\displaystyle {\bar {x}}} {\bar {x}} is the mean of the sample.
    S is the standard deviation of the sample.


Z-test

The z-score is often used in the z-test in standardized testing – the analog of the Student's
t-test for a population whose parameters are known, rather than estimated. As it is very
unusual to know the entire population, the t-test is much more widely used.
Prediction intervals
'''

'''
Comparison of scores measured on different scales: ACT and SAT

When scores are measured on different scales, they may be converted to z-scores to aid
comparison. Dietz et al.[7] give the following example comparing student scores on the SAT
and ACT high school tests. The table shows the mean and standard deviation for total score
on the SAT and ACT. Suppose that student A scored 1800 on the SAT, and student B scored 24
on the ACT. Which student performed better relative to other test-takers?


                     SAT     ACT
Mean                 1500    21
Standard deviation   300     5

The z-score for student A is z = (x − μ)/ σ = (1800 − 1500) / 300 = 1
The z-score for student B is z = (x − μ)/ σ = (24 − 21) / 5 = 0.6

Because student A has a higher z-score than student B,
student A performed better compared to other test-takers than did student B.

'''


def z_score(x, μ, σ):
    z = (x - μ) / σ
    return z


A_z_score = z_score(1800, 1500, 300)

print(A_z_score)

B_z_score = z_score(24, 21, 5)

print(B_z_score)

'''
Percentage of observations below a z-score

Continuing the example of ACT and SAT scores, if it can be further assumed that both ACT
and SAT scores are normally distributed (which is approximately correct), then the z-scores
may be used to calculate the percentage of test-takers who received lower scores than
students A and B.
'''


def z_score_to_index_column(df, z):
    z_10 = z * 10
    row = int(z_10)
    col = (z_10 - row) / 10.0
    row = str(row / 10.0)
    col = to_col(col)

    # https://goodcalculators.com/p-value-calculator/
    return [
        {'Left-tailed p-value:  P(Z < z) = ': f16(df.loc[row, col])},
        {'Right-tailed p-value: P(Z > z) = ': f16(1 - df.loc[row, col])},
        {'Two-tailed p-value: 2P(Z > |z|) = ': f16(2 * (1 - df.loc[row, col]))},

    ]


def f(z):
    return '{:1.2f}'.format(z)


def f16(z):
    return '{:1.2f}'.format(z)


if __name__ == '__main__':

    df = to_cumulative()

    pa = z_score_to_index_column(df, A_z_score)

    print([f(A_z_score), pa])

    pa = z_score_to_index_column(df, B_z_score)

    print([f(B_z_score), pa])

    for z in [1.6, 1.65, 1.96, 2.58]:
        p = z_score_to_index_column(df, z)
        print([f(z), p])
