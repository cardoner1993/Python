import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from PyNomaly import loop
from sklearn.preprocessing import StandardScaler
import os
cwd = os.getcwd()
##change directory in python.
os.chdir("C:/Users/David/TutorialPython/scripts")
#database = cwd + "/data/global.db"


class localout:
    def __init__(self,df):
        self.df = df

    def kknout(self):
        df = self.df.copy()
        scaler = StandardScaler()
        scaled = scaler.fit_transform(df.reshape(-1,1))
        df_variables = pd.DataFrame(scaled)
        df_variables.index = df.index
        m = loop.LocalOutlierProbability(df_variables, extent=0.95, n_neighbors=3).fit()
        scores = m.local_outlier_probabilities
        print(scores)
        return scores, df_variables

def iswrong(value):
    if pd.isnull(value) or value in [None, "None", "NONE", "NA", "", " ", "nan", "NaN", "-", "."]:
        value = np.nan
    return (value)

data_for_disaggregation = pd.read_csv("data_expl.csv")

##Now change structure of data.
data_test = data_for_disaggregation.head(10000).copy()
##data_test = data_for_disaggregation.copy()
data_test["yeamon"] = data_test.apply(lambda x: str(x.year)+str(x.month), axis=1)

seriemonth = data_test.pivot_table('MAINSUPPLY', index='location', columns='yeamon')
seriemonth.dropna(axis=0, how='any', thresh=None, subset=None, inplace=True)


a = seriemonth.iloc[0,:]
[s / t for s, t in zip(a, a[1:])]

#iterating over locations.
for d in range(len(seriemonth)):
    #pd.rolling_mean(seriemonth.iloc[d,:],1) #sera el dato.
    roll2 = pd.rolling_mean(seriemonth.iloc[d,:],24)
    roll5 = pd.rolling_mean(seriemonth.iloc[d, :], 7*24)
    roll10 = pd.rolling_mean(seriemonth.iloc[d, :], 7*24*4)
#     rollmed5 = pd.rolling_median(seriemonth.iloc[d,:],5)
#     ax_del = seriemonth.iloc[d, :].plot(x_compat=True, style='--',color=["r", "b"])
#     roll2.plot(color=["b"], ax=ax_del, legend=0)
#     roll5.plot(color=["g"], ax=ax_del, legend=0)
#     roll10.plot(color=["orange"], ax=ax_del, legend=0)
#     rollmed5.plot(color=["black"],ax=ax_del,legend="best")
#     #Todo think in a metric to evaluate.
#     #Todo add a legend to plot by individual.
#     plt.title('Delays per 2,5 and 10 months on consumption by location')
#     plt.xlabel('Date')
#     plt.ylabel('Months')
#     plt.show()

l = []
for d in range(len(seriemonth)):
    localoutlier = localout(seriemonth.iloc[d,:])
    p_lof, data = localoutlier.kknout()
    ax_del = data.plot(x_compat=True, style='ro')
    roll5 = pd.rolling_mean(data, 4)
    roll5.plot(color=["g"], ax=ax_del,style='--', legend=0)
    l.append(p_lof)
    print(p_lof)
    data

lofdf = pd.DataFrame(np.array(l).reshape(len(seriemonth),len(seriemonth.columns)), columns = seriemonth.columns)

lofdf[(lofdf>1).any(axis=1)]
