# get arguments from command line
import sys
import os
import pandas as pd
import pickle
import numpy as np


def main(args) -> None:
    """ Main function to be called when the script is run from the command line. 
    This function will recommend songs based on the user's input and save the
    playlist to a csv file.
    
    Parameters
    ----------
    args: list 
        list of arguments from the command line
    Returns
    -------
    None
    """
    arg_list = args[1:]
    if len(arg_list) == 0:
        print("Usage: python3 musicRecommender.py <csv file>")
        sys.exit()
    else:
        file_name = arg_list[0]
        if not os.path.isfile(file_name):
            print("File does not exist")
            sys.exit()
        else:
            userPreferences = pd.read_csv(file_name)

    # this code is just to check, delete later.
    print(userPreferences.head())

    cluster_module, clustered_indexes, pca_module = pickle.load(open("model.pkl", "rb"))

    df_new=scale_row(pd.read_csv(args[1]))
    playlist_cluster=cluster_module.predict(pca_module.transform(df_new.sample(5)))
    same_cluster = np.array(clustered_indexes)[np.isin(cluster_module.labels_, playlist_cluster)]
    same_cluster=[str(item) for item in same_cluster.astype(str)]
    np.random.shuffle(same_cluster)

    # TODO:
    # 1. Use your train model to make recommendations for the user.
    # 2. Output the recommendations as 5 different playlists with
    #    the top 5 songs in each playlist. (5 playlists x 5 songs)
    # 2.1. Musics in a single playlist should be from the same cluster.
    # 2.2. Save playlists to a csv file.
    # 3. Output another single playlist recommendation with all top songs from all clusters.

    np.savetxt("recs.csv", np.array([same_cluster[i:i+5] for i in range(5)]), delimiter=",", fmt='%s')

def scale_row(df):
    import math

    df_s = pd.read_csv("src_data/genres_v2.csv").drop_duplicates()
    df_s.set_index('id', inplace=True)
    df_s.drop(['Unnamed: 0', 'title', 'analysis_url', 'track_href', 'uri',
             'type', 'song_name', 'genre'], axis=1, inplace=True)
    perc = np.array([98.5, 1.5])
    df_mm = df_s.describe(percentiles=perc / 100)
    df_mm = [df_mm.loc[str(perc[1]) + '%', :],
             df_mm.loc[str(perc[0]) + '%', :] - df_mm.loc[str(perc[1]) + '%', :]]

    df.set_index('id', inplace=True)
    df_genre = df['genre']
    df.drop(['Unnamed: 0', 'title', 'analysis_url', 'track_href', 'uri',
             'type', 'song_name', 'genre'], axis=1, inplace=True)
    df = df.apply(lambda col: (col - df_mm[0][col.name]) / df_mm[1][col.name]) - 0.5

    max_v, min_v = 0.5, -0.5
    df = df.apply(lambda col: col.apply(lambda v: v if np.isscalar(v) and max_v >= v >= min_v
    else np.sign(v) * math.log(abs(v * 8) + 1, 5) / 2))

    df['genre'] = df_genre
    del df_mm, perc

    df['genre']=df['genre'].astype('category').cat.codes
    df['genre']/=max(df['genre'])
    df['genre']-=0.5
    df['genre'].describe()

    df_new = df.copy()
    df_new.drop(['mode', 'time_signature', 'key'], axis=1, inplace=True)
    f = lambda v: np.sign(v) * math.log(abs(v * 16) + 1, 5) / 1.76
    df_new['danceability'] = df['danceability'] * 0.95
    df_new['speechiness'] = -0.5 + (df['speechiness'] + 0.5).apply(f)
    df_new['acousticness'] = -0.25 + (df['acousticness'] + 0.5).apply(f).apply(f) * 1.1 - 0.3
    df_new['instrumentalness'] = -0.5 + (df['instrumentalness'] + 0.5).apply(f).apply(f)
    df_new['liveness'] = -0.5 + (df['liveness'] + 0.5).apply(f)
    df_new['duration_ms'] = df['duration_ms'] * 0.8 + 0.15
    df_new = df_new.apply(lambda col: col.apply(lambda v: v if np.isscalar(v) and max_v >= v >= min_v
    else np.sign(v) * math.log(abs(v * 8) + 1, 5) / 2))

    return df_new


if __name__ == "__main__":
    # get arguments from command line
    args = sys.argv
    main(args)