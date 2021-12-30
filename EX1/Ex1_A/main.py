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
        "host_since",
        "host_response_time",
        "host_response_rate",
        "host_acceptance_rate",
        "host_is_superhost",
        "host_neighbourhood",
        "host_total_listings_count",
        "host_verifications",
        "host_has_profile_pic",
        "host_identity_verified",
        "neighbourhood_cleansed",
        "neighbourhood_group_cleansed",
        "latitude",
        "longitude",
        "room_type",
        "accommodates",
        "bathrooms_text",
        "bedrooms",
        "beds",
        "amenities",
        "price",
        "minimum_nights",
        "maximum_nights",
        "minimum_nights_avg_ntm",
        "maximum_nights_avg_ntm",
        "has_availability",
        "availability_30",
        "availability_90",
        "availability_365",
        "number_of_reviews",
        "number_of_reviews_ltm",
        "review_scores_rating",
        "review_scores_accuracy",
        "review_scores_cleanliness",
        "review_scores_checkin",
        "review_scores_communication",
        "review_scores_location",
        "review_scores_value",
        "instant_bookable"
    ]


def separate_cols(df=True, save=False, reduce=False):
    if df is True: df = pd.read_csv(r"src_data\listings.csv")[get_cols()]

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

    def array_cleaner(df_l, lim=400):
        l = {}
        for ams in df_l:
            ams = eval(ams)
            if ams is None: continue
            for am in ams:
                try: l[am] += 1
                except: l[am] = 1
        l = dict(sorted(l.items(), key=lambda item: item[1], reverse=False))

        l = {k: l[k] for k in l.keys() if l[k] > lim}
        l = list(l.keys())
        return lambda x: [k for k in eval(x) if k in l] if type(x) is str and eval(x) is not None else x

    def value_split_in_range(a,b,t):
        delta = (b-a)/t
        return lambda x: (round(int(x/delta)*delta,5) if not math.isnan(x) else x) \
            if a < x < b else (round(a-delta*2,5) if a > x else round(b+delta*2,5))

    df["host_since"] = df["host_since"].apply(to_year_fraction)
    df["host_response_rate"] = df["host_response_rate"].apply(lambda x: float(x[:-1])/100 if type(x) is str else x)
    df["host_acceptance_rate"] = df["host_acceptance_rate"].apply(lambda x: float(x[:-1])/100 if type(x) is str else x)
    df["host_is_superhost"] = df["host_is_superhost"].apply(lambda x: 0 if x == "f" else 1)
    df["host_has_profile_pic"] = df["host_has_profile_pic"].apply(lambda x: 0 if x == "f" else 1)
    df["host_identity_verified"] = df["host_identity_verified"].apply(lambda x: 0 if x == "f" else 1)
    df["price"] = df["price"].apply(lambda x: float(x[1:].replace(",", "")) if type(x) is str else x)
    df["instant_bookable"] = df["instant_bookable"].apply(lambda x: 0 if x == "f" else 1)

    for col in df.columns:
        if "review_score" in col:
            df[col] = df[col].apply(lambda x: x/5 if not math.isnan(x) else x)

    if reduce:
        df["host_since"] = df["host_since"].apply(floor_fractions_to_ith(4))
        df["host_response_rate"] = df["host_response_rate"].apply(floor_fractions_to_ith(20))
        df["host_acceptance_rate"] = df["host_acceptance_rate"].apply(floor_fractions_to_ith(20))
        df["price"] = df["price"].apply(lambda x: int(x/20)*20 if not math.isnan(x) else x)
        df["amenities"] = df["amenities"].apply(array_cleaner(df["amenities"]))
        df["host_verifications"] = df["host_verifications"].apply(array_cleaner(df["host_verifications"]), 200)
        df["latitude"] = df["latitude"].apply(value_split_in_range(40.6, 40.85, 20))
        df["longitude"] = df["longitude"].apply(value_split_in_range(-74.02, -73.8, 20))
        df["number_of_reviews"] = df["number_of_reviews"].apply(lambda x: (int(x/5)*5 if not math.isnan(x) else x) if x>10 else x)
        for col in df.columns:
            if "review_score" in col:
                df[col] = df[col].apply(floor_fractions_to_ith(8))
            if "nights" in col or "availability_" in col:
                df[col] = df[col].apply(lambda x: (int(x/5)*5 if not math.isnan(x) else x) if x>12 else int(x/3)*3)
    if save: df.to_csv(r"working_data\separated_cols.csv", index=False)

    return df


def count_unique(df=True, save=False):
    if df is True: df = pd.read_csv(r"working_data\separated_cols.csv")

    def array_cleaner(df_l):
        l = {}
        for ams in df_l:
            if type(ams) is str: ams = eval(ams)
            if ams is None:
                try: l["None"] += 1
                except: l["None"] = 1
                continue
            for am in ams:
                try: l[am] += 1
                except: l[am] = 1
        l = dict(sorted(l.items(), key=lambda item: item[1], reverse=True))
        return l

    count = {col: list(zip(df[col].value_counts().index, df[col].value_counts())) for col in df.columns}

    count["amenities"]=list(array_cleaner(df["amenities"]).items())
    count["host_verifications"]=list(array_cleaner(df["host_verifications"]).items())

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

    cols=get_cols()
    for col in ["host_total_listings_count","host_verifications",
                "host_neighbourhood","neighbourhood_cleansed",
                "amenities","bathrooms_text","minimum_nights",
                "maximum_nights","minimum_nights_avg_ntm","maximum_nights_avg_ntm",]:
        cols.remove(col)

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
    count_unique(separate_cols(reduce=True), save=True)
    # mean_chart(save=True)
    # normality_chart(save=True)
    # plot_counts()
