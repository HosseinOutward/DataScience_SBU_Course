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
        "DATE_REPORTED",
        "DATE_OCCURED",
        "UOR_DESC",
        "CRIME_TYPE",
        "NIBRS_CODE",
        "UCR_HIERARCHY",
        "ATT_COMP",
        "LMPD_DIVISION",
        "PREMISE_TYPE",
        "BLOCK_ADDRESS",
        "CITY",
        "ZIP_CODE",
    ]


def separate_cols(df=True, save=False, reduce=False):
    if df: df = pd.read_csv(r"src_data\Crime_Data_2019.csv")[get_cols()]

    def to_year_fraction(date):
        def s(date): return time.mktime(date.timetuple())

        if type(date) is not str: return date

        date = dt.strptime(date.split(" ")[0], '%Y-%m-%d')
        year = date.year
        startOfThisYear = dt(year=year, month=1, day=1)
        startOfNextYear = dt(year=year + 1, month=1, day=1)
        if year<1970: return year
        yearElapsed = s(date) - s(startOfThisYear)
        yearDuration = s(startOfNextYear) - s(startOfThisYear)
        fraction = yearElapsed / yearDuration

        return date.year + fraction

    def to_time_of_day(date):
        if type(date) is not str: return date
        date = dt.strptime(date.split(" ")[1], '%H:%M:%S')
        fraction = (date.hour*60*60+date.minute*60+date.second) / 86400
        return fraction

    def floor_fractions_to_ith(i): return lambda x: int(x)+int((x-int(x))*i)/i if not math.isnan(x) else x

    def turn_number(x):
        if type(x) is str:
            try: x=eval(x)
            except: x=None
        return int(x) if x is not None and not math.isnan(x) else x

    df["Time_reported"] = df["DATE_REPORTED"].apply(to_time_of_day)
    df["Time_occurred"] = df["DATE_OCCURED"].apply(to_time_of_day)
    df["DATE_REPORTED"] = df["DATE_REPORTED"].apply(to_year_fraction)
    df["DATE_OCCURED"] = df["DATE_OCCURED"].apply(to_year_fraction)
    df["ZIP_CODE"] = df["ZIP_CODE"].apply(turn_number)

    if reduce:
        df["DATE_REPORTED"] = df["DATE_REPORTED"].apply(floor_fractions_to_ith(4))
        df["DATE_OCCURED"] = df["DATE_OCCURED"].apply(floor_fractions_to_ith(4))
        df["Time_reported"] = df["Time_reported"].apply(floor_fractions_to_ith(20))
        df["Time_occurred"] = df["Time_occurred"].apply(floor_fractions_to_ith(20))
    if save: df.to_csv(r"working_data\separated_cols.csv", index=False)

    return df


def count_unique(df=True, save=False):
    if df is True: df = pd.read_csv(r"working_data\separated_cols.csv")

    count = {col: list(zip(df[col].value_counts().index, df[col].value_counts())) for col in df.columns}

    count_df=pd.DataFrame()
    for col in count.keys():
        col_df = pd.DataFrame({col:count[col]})
        count_df = pd.concat([count_df, col_df], axis=1)

    if save: count_df.to_csv(save+r"\unique_counts.csv", index=False)

    return count_df


def mean_chart(df=True, save=False):
    if df is True: df = pd.read_csv(r"working_data\separated_cols.csv")

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

    if save: pd.DataFrame(mean_chart).to_csv(save+r"\mean_chart.csv", index=False)

    return df


def normality_chart(df=True, save=False):
    if df is True: df = pd.read_csv(r"working_data\separated_cols.csv")

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

    if save: pd.DataFrame(normality_chart).to_csv(save+r"\normality_chart.csv", index=False)

    return df


def plot_counts(df=True, reduced=True, save="working_data"):
    if df is True: df = pd.read_csv(r"working_data\separated_cols.csv")
    if reduced: df_mean = pd.read_csv(r"working_data\mean_chart.csv")

    cols=get_cols()
    for col in ["UOR_DESC", "BLOCK_ADDRESS"]:
        cols.remove(col)
    cols=["Time_reported","Time_occurred"]+cols

    for col in ["ZIP_CODE"]:
        print(col)
        pop = df[col][df[col].notna()]
        plt.clf()
        if col=="ZIP_CODE":
            plt.bar(pop.value_counts().index, pop.value_counts())
            plt.xlim(38000,42000)
        else: sns.histplot(pop)
        plt.xticks(rotation=45)
        plt.subplots_adjust(bottom=0.2)
        plt.title(col)
        plt.savefig(save+"\\hist_plot\\all\\"+col+"_all.png")

        if reduced:
            m,v=0,0
            if df_mean[col][df_mean[col].notna()].empty: continue
            for a in df_mean[col]: m,v=tuple(map(sum, zip((m,v), eval(a))))
            m,v = m/len(df_mean[col]), v/len(df_mean[col])*1.1
            plt.xlim(max(m-v, min(pop)),min(m+v, max(pop)))
            if col=="ZIP_CODE": plt.xlim(40190,40310)
            plt.savefig(save+"\\hist_plot\\reduced\\"+col+"_reduced.png")


def with_without(test="df_base['CRIME_TYPE'] == ASSAULT", val='only_ASSAULT'):
    df_base = pd.read_csv(r"working_data\separated_cols.csv")
    df_base_r = separate_cols(reduce=True)

    df=df_base.loc[eval(test)]
    df_r=df_base_r.loc[eval(test)]
    df.to_csv("working_data\\seperated\\"+val+"\\separated_cols.csv", index=False)
    count_unique(df_r, save="working_data\\seperated\\"+val)
    mean_chart(df, save="working_data\\seperated\\"+val)
    normality_chart(df, save="working_data\\seperated\\"+val)
    plot_counts(df, save="working_data\\seperated\\"+val)


if __name__ == '__main__':
    separate_cols(save="working_data")
    count_unique(separate_cols(reduce=True), save="working_data")
    # mean_chart(save="working_data")
    # normality_chart(save="working_data")
    # plot_counts()
    # with_without(test="round(df_base['DATE_OCCURED'], 0) > 2015", val='2021')