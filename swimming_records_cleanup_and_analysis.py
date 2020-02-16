# http://gijskoot.nl/pandas/swimming/sports/records/2018/01/16/swimming-records.html

import pandas as pd
import matplotlib.pyplot as plt
# import statsmodels.api as sm
# import matplotlib.pylab as plt
# from pandas.plotting import andrews_curves

url = "name/[Printer Friendly]records.html"

tables = pd.read_html(url, header=1, encoding='utf-8')

df = tables[0].dropna(axis=0, how='all').dropna(axis=1, how='all')

pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('max_colwidth', -1)

to_int = {'Pos': 'int', 'Pts': 'int', 'EventAgeCurrent': 'int', }
keys = list(to_int.keys())

# df[['Pts']] = df[['Pts']].fillna(value=0)
df[keys] = df[keys].fillna(value=0).astype('int')
# df.fillna({'Pts': 0, })

for i, e in enumerate(df['EventAgeCurrent']):
    if e < 100:
        df['EventAgeCurrent'][i] = e / 10
    if e < 10000:
        df['EventAgeCurrent'][i] = e / 10

for i, e in enumerate(df['Event']):
    df['Event'][i] = e.split(')')[1]

parsed_Finals = df.Finals.str.extract("(?P<m>\d{1,2})?:?(?P<s>\d{2})\.(?P<ms>\d{2})*", expand=True)

# print(parsed_Finals)

Finals_in_seconds = parsed_Finals.m.astype(float).fillna(0) * 60 + parsed_Finals.s.astype(float) + parsed_Finals.ms.astype(float) / 100

df.Finals = Finals_in_seconds

print(df)
print()

df['Date ofSwim'] = pd.DatetimeIndex(data=df['Date ofSwim'])
df = df.sort_values(by=['Event', 'Date ofSwim'], ascending=True,)

print(df)

# https://stackoverflow.com/questions/43707620/plotting-a-time-series


def plot_50_Free():
    dfg = df.loc[df['Event'] == '50 Free']
#     dfg['Date ofSwim'] = pd.DatetimeIndex(data=dfg['Date ofSwim'])
#     dfg = dfg.sort_values(by=['Event', 'Date ofSwim'], ascending=True,)

#     print(dfg)
    dfg.plot(x='Date ofSwim', y='Finals', label='50 Free')


def plot_event(event='50 Free'):
    dfg = df.loc[df['Event'] == event]
#     dfg['Date ofSwim'] = pd.DatetimeIndex(data=dfg['Date ofSwim'])
#     dfg = dfg.sort_values(by=['Event', 'Date ofSwim'], ascending=True,)
#     print(dfg)
    ax = dfg.plot(x='Date ofSwim', y='Finals', marker='.', label=event)
    ax.set_xlabel("Date")
    ax.set_ylabel("Finals (in seconds)")

    # ax.set(xlabel="x label", ylabel="y label")

#     print(x)

# plot_event()
# plot_event('100 Fly')


print(["df['Event']: ", df['Event']])

for e in set(df['Event']):
    plot_event(e)


def plot_df_pivot_table():
    # https://stackoverflow.com/questions/38197964/pandas-plot-multiple-time-series-dataframe-into-a-single-plot
    df.pivot_table(index='Date ofSwim', columns='Event', values='Finals').plot()

# plot_df_pivot_table()


# multiple line plot
def plot_multiple_line():
    # Valid font size are xx-small, x-small, small, medium, large, x-large, xx-large, larger, smaller, None
    plt.xticks(
        rotation=45,
        horizontalalignment='right',
        fontweight='light',
        fontsize='medium',  # 'x-large'
    )

    plt.xlabel('Date')
    plt.ylabel('Seconds')

    for e in set(df.Event):
        dfg = df.loc[df['Event'] == e]
        plt.plot(dfg['Date ofSwim'], dfg['Finals'], label=e)
        plt.legend()
#         plt.xlabel(ax.get_xlabel(), rotation=90)


plot_multiple_line()

plt.show()
