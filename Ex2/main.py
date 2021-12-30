import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime as dt
import time
import math
import numpy as np
import scipy.stats as stats
import seaborn as sns


def get_cols():
    return [

    ]


def separate_cols(df=True, save=False, reduce=False):
    if df is True: df = pd.read_csv(r"src_data\owid-covid-data.csv")

    def to_year_fraction(date):
        def sinceEpoch(date): return time.mktime(date.timetuple())

        s = sinceEpoch

        if type(date) is not str: return date

        date = dt.strptime(date, '%Y-%m-%d')
        year = date.year
        startOfThisYear = dt(year=year, month=1, day=1)
        startOfNextYear = dt(year=year + 1, month=1, day=1)

        yearElapsed = s(date) - s(startOfThisYear)
        yearDuration = s(startOfNextYear) - s(startOfThisYear)
        fraction = yearElapsed / yearDuration

        return date.year + fraction

    def floor_fractions_to_ith(i): return lambda x: int(x)+int((x-int(x))*i)/i if not math.isnan(x) else x

    # def value_split_in_range(a,b,t):
    #     delta = (b-a)/t
    #     return lambda x: (round(int(x/delta)*delta,5) if not math.isnan(x) else x) \
    #         if a < x < b else (round(a-delta*2,5) if a > x else round(b+delta*2,5))

    df["date"] = df["date"].apply(to_year_fraction)

    if reduce:
        df["date"] = df["date"].apply(floor_fractions_to_ith(4))
    if save: df.to_csv(r"working_data\separated_cols.csv", index=False)

    return df


def count_unique(df=True, save=False):
    if df is True: df = pd.read_csv(r"working_data\separated_cols.csv")

    count = {col: list(zip(df[col].value_counts().index, df[col].value_counts())) for col in df.columns}

    count_df=pd.DataFrame()
    for col in count.keys():
        col_df = pd.DataFrame({col:count[col]})
        count_df = pd.concat([count_df, col_df], axis=1)

    if save: count_df.to_csv(r"working_data\unique_counts.csv", index=False)

    return count_df


def mean_chart(df=True, save=False):
    if df: df = pd.read_csv(r"working_data\separated_cols.csv")

    mean_chart=pd.DataFrame()
    for col in df.columns:
        mean_df = {}
        try:
            pop = df[col][df[col].notna()]
            mean_df[col]=[]
            for s in range(10000):
                sample = np.random.choice(pop, 250)
                mean_df[col].append((sample.mean(), sample.std()))
        except: pass
        mean_chart = pd.concat([mean_chart, pd.DataFrame(mean_df)], axis=1)

    if save: pd.DataFrame(mean_chart).to_csv(r"working_data\mean_chart.csv", index=False)

    return df


def normality_chart(df=True, save=False):
    if df: df = pd.read_csv(r"working_data\separated_cols.csv")

    # df.drop([
    # ], axis=1, inplace=True)

    normality_chart = pd.DataFrame()
    for col in df.columns:
        print(col)
        normality_df = {}
        try:
            pop = df[col][df[col].notna()]
            normality_df[col]=[]
            for s in range(1000):
                sample = np.random.choice(pop, 250)
                z,p = stats.normaltest(sample)
                normality_df[col].append((z,p))
        except: pass
        normality_chart = pd.concat([normality_chart, pd.DataFrame(normality_df)], axis=1)

    if save: pd.DataFrame(normality_chart).to_csv(r"working_data\normality_chart.csv", index=False)

    return df


def plot_counts(df=True, reduced=True):
    if df: df = pd.read_csv(r"working_data\separated_cols.csv")
    if reduced: df_mean = pd.read_csv(r"working_data\mean_chart.csv")

    cols=df.columns

    # df.drop([
    # ], axis=1, inplace=True)

    for col in cols:
        print(col)
        pop = df[col][df[col].notna()]
        plt.clf()
        sns.histplot(pop)
        plt.title(col)
        plt.savefig("working_data\\hist_plot\\all\\"+col+"_all.png")

        if reduced:
            m,v=0,0
            if df_mean[col][df_mean[col].notna()].empty: continue
            for a in df_mean[col]: m,v=tuple(map(sum, zip((m,v), eval(a))))
            m,v = m/len(df_mean[col]), v/len(df_mean[col])*1.1
            plt.xlim(max(m-v, min(pop)),min(m+v, max(pop)))
            plt.savefig("working_data\\hist_plot\\reduced\\"+col+"_reduced.png")


def with_without(col="CRIME_TYPE", val1='ASSAULT', val='ASSAULT'):
    df_base = pd.read_csv(r"working_data\separated_cols.csv")
    df_base_r = pd.read_csv(r"working_data\separated_cols.csv")

    df=df_base.loc[df_base[col] == val1]
    df_r=df_base_r.loc[df_base_r[col] == val1]
    df.to_csv("working_data\\seperated\\only_"+val+"\\separated_cols.csv", index=False)
    count_unique(df_r, save="working_data\\seperated\\only_"+val)
    mean_chart(df, save="working_data\\seperated\\only_"+val)
    normality_chart(df, save="working_data\\seperated\\only_"+val)
    plot_counts(df, save="working_data\\seperated\\only_"+val)

    df=df_base.loc[df_base[col] != val1]
    df_r=df_base_r.loc[df_base_r[col] != val1]
    df.to_csv("working_data\\seperated\\without_"+val+"\\separated_cols.csv", index=False)
    count_unique(df_r, save="working_data\\seperated\\without_"+val)
    mean_chart(df, save="working_data\\seperated\\without_"+val)
    normality_chart(df, save="working_data\\seperated\\without_"+val)
    plot_counts(df, save="working_data\\seperated\\without_"+val)


if __name__ == '__main__':
    # separate_cols(save=True)
    # count_unique(separate_cols(reduce=True), save=True)
    # mean_chart(save=True)
    # normality_chart(save=True)
    plot_counts()
