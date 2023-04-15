import streamlit as st
# Basic Streamlit Settings
st.set_page_config(page_title='Music Recommendation', layout = 'wide', initial_sidebar_state = 'auto')
from streamlit_option_menu import option_menu
import streamlit.components.v1 as components
import plotly.express as px
from sklearn.neighbors import NearestNeighbors
import pandas as pd
import numpy as np
from load_css import local_css
from PIL import Image
import pydeck as pdk
import plotly.figure_factory as ff
import base64
import streamlit.components.v1 as components
import webbrowser


import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
from imblearn.over_sampling import SMOTE

import time
import os
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
import random
from functools import reduce
import spotipy
import spotipy.util as util
from spotipy.oauth2 import SpotifyClientCredentials
from spotipy import oauth2



import sklearn
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn import metrics 
from sklearn.metrics import f1_score
# Models
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression 
from sklearn.tree import DecisionTreeClassifier

local_css("style.css")



cid = 'bb34b741629d4173bf4bffa6417a2270'
secret = 'c0c94901909e42e483aac0680375c441'
redirect_uri='http://localhost:7777/callback'
username = 'y1spdovdjdvlieulko3su8e2f'

scope = 'user-top-read playlist-modify-private playlist-modify-public'
token = util.prompt_for_user_token(username, scope, client_id=cid, client_secret=secret, redirect_uri=redirect_uri)

if token:
    sp = spotipy.Spotify(auth=token)
else:
    print("Can't get token for", username)

def fetch_audio_features(sp, df):
    playlist = df[['track_id','track_name']] 
    index = 0
    audio_features = []
    
    while index < playlist.shape[0]:
        audio_features += sp.audio_features(playlist.iloc[index:index + 50, 0])
        index += 50
    
    features_list = []
    for features in audio_features:
        features_list.append([features['danceability'],
                              features['acousticness'],
                              features['energy'], 
                              features['tempo'],
                              features['instrumentalness'], 
                              features['loudness'],
                              features['liveness'],
                              features['duration_ms'],
                              features['key'],
                              features['valence'],
                              features['speechiness'],
                              features['mode']
                             ])
    
    df_audio_features = pd.DataFrame(features_list, columns=['danceability', 'acousticness', 'energy','tempo', 
                                                             'instrumentalness', 'loudness', 'liveness','duration_ms', 'key',
                                                             'valence', 'speechiness', 'mode'])
    
    df_playlist_audio_features = pd.concat([playlist, df_audio_features], axis=1)
    df_playlist_audio_features.set_index('track_name', inplace=True, drop=True)
    return df_playlist_audio_features

def getTrackIDs(playlist_id):
    playlist = sp.user_playlist('spotify', playlist_id)
    for item in playlist['tracks']['items'][:50]:
        track = item['track']
        ids.append(track['id'])
    return

# Creating a function get features of each track from track id
def getTrackFeatures(track_id):
  meta = sp.track(track_id)
  features = sp.audio_features(track_id)

  # meta
  track_id = track_id
  name = meta['name']
  album = meta['album']['name']
  artist = meta['album']['artists'][0]['name']
  release_date = meta['album']['release_date']
  length = meta['duration_ms']
  popularity = meta['popularity']

  # features
  acousticness = features[0]['acousticness']
  danceability = features[0]['danceability']
  energy = features[0]['energy']
  instrumentalness = features[0]['instrumentalness']
  liveness = features[0]['liveness']
  loudness = features[0]['loudness']
  speechiness = features[0]['speechiness']
  tempo = features[0]['tempo']
  time_signature = features[0]['time_signature']

  track = [track_id, name, album, artist, release_date, length, popularity, danceability, acousticness, energy, instrumentalness, liveness, loudness, speechiness, tempo, time_signature]
  return track


def spr_sidebar():
    with st.sidebar:
        st.info('**Music Recommendation**')
        home_button = st.button("About Us")
        rec_button = st.button('Recommendation Engine')
        random_recom = st.button("Personalized Recommendation")
        
        
        st.session_state.log_holder = st.empty()
        if home_button:
            st.session_state.app_mode = 'home'
        
        if rec_button:
            st.session_state.app_mode = 'recommend'
        
        if random_recom:
            st.session_state.app_mode = 'recommend_rand'
        






    



def load_data():
    df = pd.read_csv(
        "filtered_track_df.csv")
    df['genres'] = df.genres.apply(
        lambda x: [i[1:-1] for i in str(x)[1:-1].split(", ")])
    exploded_track_df = df.explode("genres")
    return exploded_track_df


genre_names = ['Dance Pop', 'Electronic', 'Electropop', 'Hip Hop',
               'Jazz', 'K-pop', 'Latin', 'Pop', 'Pop Rap', 'R&B', 'Rock']
audio_feats = ["acousticness", "danceability",
               "energy", "instrumentalness", "valence", "tempo"]

exploded_track_df = load_data()

genre_names = ['Dance Pop', 'Electronic', 'Electropop', 'Hip Hop',
               'Jazz', 'K-pop', 'Latin', 'Pop', 'Pop Rap', 'R&B', 'Rock']
audio_feats = ["acousticness", "danceability",
               "energy", "instrumentalness", "valence", "tempo"]

def n_neighbors_uri_audio(genre, start_year, end_year, test_feat):
    genre = genre.lower()
    genre_data = exploded_track_df[(exploded_track_df["genres"] == genre) & (
        exploded_track_df["release_year"] >= start_year) & (exploded_track_df["release_year"] <= end_year)]
    genre_data = genre_data.sort_values(by='popularity', ascending=False)[:500]

    neigh = NearestNeighbors()
    neigh.fit(genre_data[audio_feats].to_numpy())

    n_neighbors = neigh.kneighbors([test_feat], n_neighbors=len(
        genre_data), return_distance=False)[0]

    uris = genre_data.iloc[n_neighbors]["uri"].tolist()
    audios = genre_data.iloc[n_neighbors][audio_feats].to_numpy()
    return uris, audios



def rec_page():
    
    st.header("RECOMMENDATION ENGINE")
   
    with st.container():
        col1, col2, col3, col4 = st.columns((2, 0.5, 0.5, 0.5))
    with col3:
        st.markdown("***Choose your genre:***")
        genre = st.radio(
            "",
            genre_names, index=genre_names.index("Pop"))
    with col1:
        st.markdown("***Choose features to customize:***")
        start_year, end_year = st.slider(
            'Select the year range',
            1990, 2019, (2015, 2019)
        )
        
        acousticness = st.selectbox(
            'Acousticness',
            (0.0, 1.0, 0.5))
        danceability = st.slider(
            'Danceability',
            0.0, 1.0, 0.5)
        energy = st.slider(
            'Energy',
            0.0, 1.0, 0.5)
        instrumentalness = st.slider(
            'Instrumentalness',
            0.0, 1.0, 0.0)
        valence = st.slider(
            'Valence',
            0.0, 1.0, 0.45)
        tempo = st.slider(
            'Tempo',
            0.0, 244.0, 118.0)
        tracks_per_page = 12
        test_feat = [acousticness, danceability,
                     energy, instrumentalness, valence, tempo]
        uris, audios = n_neighbors_uri_audio(
            genre, start_year, end_year, test_feat)
        tracks = []
        for uri in uris:
            track = """<iframe src="https://open.spotify.com/embed/track/{}" width="260" height="380" frameborder="0" allowtransparency="true" allow="encrypted-media"></iframe>""".format(
                uri)
            tracks.append(track)
    if 'previous_inputs' not in st.session_state:
        st.session_state['previous_inputs'] = [
            genre, start_year, end_year] + test_feat
    current_inputs = [genre, start_year, end_year] + test_feat

    if current_inputs != st.session_state['previous_inputs']:
        if 'start_track_i' in st.session_state:
            st.session_state['start_track_i'] = 0
    st.session_state['previous_inputs'] = current_inputs

    if 'start_track_i' not in st.session_state:
        st.session_state['start_track_i'] = 0

    with st.container():
        col1, col2, col3 = st.columns([2, 1, 2])
    if st.button("Recommend More Songs"):
        if st.session_state['start_track_i'] < len(tracks):
            st.session_state['start_track_i'] += tracks_per_page

    current_tracks = tracks[st.session_state['start_track_i']
        : st.session_state['start_track_i'] + tracks_per_page]
    current_audios = audios[st.session_state['start_track_i']
        : st.session_state['start_track_i'] + tracks_per_page]
    if st.session_state['start_track_i'] < len(tracks):
        for i, (track, audio) in enumerate(zip(current_tracks, current_audios)):
            if i % 2 == 0:
                with col1:
                    components.html(
                        track,
                        height=400,
                    )
                    with st.expander("See more details"):
                        df = pd.DataFrame(dict(
                            r=audio[:5],
                            theta=audio_feats[:5]))
                        fig = px.line_polar(
                            df, r='r', theta='theta', line_close=True)
                        fig.update_layout(height=400, width=340)
                        st.plotly_chart(fig)

            else:
                with col3:
                    components.html(
                        track,
                        height=400,
                    )
                    with st.expander("See more details"):
                        df = pd.DataFrame(dict(
                            r=audio[:5],
                            theta=audio_feats[:5]))
                        fig = px.line_polar(
                            df, r='r', theta='theta', line_close=True)
                        fig.update_layout(height=400, width=340)
                        st.plotly_chart(fig)

    else:
        st.write("No songs left to recommend")
    

    


def home_page():
    st.subheader('')
    
    
    col1, col2 = st.columns(2)

def enrich_playlist(sp, username, playlist_id, playlist_tracks):
    index = 0
    results = []
    
    while index < len(playlist_tracks):
        results += sp.user_playlist_add_tracks(username, playlist_id, tracks = playlist_tracks[index:index + 50])
        index += 50

def create_playlist(sp, username, playlist_name, playlist_description):
        playlists = sp.user_playlist_create(username, playlist_name, description = playlist_description)

def fetch_playlists(sp, username):
    """
    Returns the user's playlists.
    """
        
    id = []
    name = []
    num_tracks = []
    
    # Make the API request
    playlists = sp.user_playlists(username)
    for playlist in playlists['items']:
        id.append(playlist['id'])
        name.append(playlist['name'])
        num_tracks.append(playlist['tracks']['total'])

    # Create the final df   
    df_playlists = pd.DataFrame({"id":id, "name": name, "#tracks": num_tracks})
    return df_playlists


def rand_rec():
    # Insert your Spotify username and the credentials that you obtained from spotify developer
    cid = 'bb34b741629d4173bf4bffa6417a2270'
    secret = 'c0c94901909e42e483aac0680375c441'
    redirect_uri='http://localhost:7777/callback'
    username = 'y1spdovdjdvlieulko3su8e2f'

    scope = 'user-top-read playlist-modify-private playlist-modify-public'
    token = util.prompt_for_user_token(username, scope, client_id=cid, client_secret=secret, redirect_uri=redirect_uri)

    if token:
        sp = spotipy.Spotify(auth=token)
    else:
        print("Can't get token for", username)



    df = pd.read_csv('data/playlist_songs.csv')
    df = df.drop_duplicates(subset=['track_id'])
    results = sp.current_user_top_tracks(limit=1000, offset=0,time_range='short_term')
    # Convert it to Dataframe
    track_name = []
    track_id = []
    artist = []
    album = []
    duration = []
    popularity = []
    for i, items in enumerate(results['items']):
        track_name.append(items['name'])
        track_id.append(items['id'])
        artist.append(items["artists"][0]["name"])
        duration.append(items["duration_ms"])
        album.append(items["album"]["name"])
        popularity.append(items["popularity"])

    df_favourite = pd.DataFrame({ "track_name": track_name, 
    "album": album, 
    "track_id": track_id,
    "artist": artist, 
    "duration": duration, 
    "popularity": popularity})


    fav_tracks = []
    for track in df_favourite['track_id']:
        try:  
            track = getTrackFeatures(track)
            fav_tracks.append(track)
        except:
            pass
    df_fav = pd.DataFrame(fav_tracks, columns = ['track_id', 'name', 'album', 'artist', 'release_date', 'length', 'popularity', 'danceability', 'acousticness', 'energy', 'instrumentalness', 'liveness', 'loudness', 'speechiness', 'tempo', 'time_signature'])
    df_fav = df_fav.drop(columns=['name', 'album', 'artist', 'release_date'])
    df_fav['favorite'] = 1
    df['favorite'] = 0 
    combined = pd.concat([df, df_fav])
    df_fav = combined.loc[combined['favorite'] == 1]
    df.to_csv('encoded_playlist_songs.csv', index=False)
    df_fav.to_csv('favorite_songs.csv', index=False)


    df = pd.read_csv('encoded_playlist_songs.csv')
    df_fav = pd.read_csv('favorite_songs.csv')
    df = pd.concat([df, df_fav], axis=0)

    shuffle_df = df.sample(frac=1)
    train_size = int(0.8 * len(df))

    train_set = shuffle_df[:train_size]
    test_set = shuffle_df[train_size:]

    train_set=train_set.drop(columns=['name','album','artist','release_date'])
    test_set=test_set.drop(columns=['name','album','artist','release_date'])


    X = train_set.drop(columns=['favorite', 'track_id'])
    y = train_set.favorite



    from imblearn.over_sampling import RandomOverSampler

    oversample = RandomOverSampler()
    X_train, y_train = oversample.fit_resample(X, y) 

    X_test = test_set.drop(columns=['favorite', 'track_id'])
    
    y_test = test_set['favorite']

    lr = LogisticRegression(solver='lbfgs', max_iter=400).fit(X_train, y_train)
    lr_scores = cross_val_score(lr, X_train, y_train, cv=10, scoring="f1")
    print(np.mean(lr_scores))
    lr_preds = lr.predict(X_train)
    

    parameters = {
    'max_depth':[3, 4, 5, 6, 10, 15,20,30],}
    dtc = Pipeline([('CV',GridSearchCV(DecisionTreeClassifier(), parameters, cv = 5))])
    dtc.fit(X_train, y_train)
    dt = DecisionTreeClassifier(max_depth=30).fit(X_train, y_train)
    dt_scores = cross_val_score(dt, X_train, y_train, cv=10, scoring="f1")    

    parameters = {
    'max_depth':[3, 6,12,15,20],
    'n_estimators':[10, 20,30]}
    clf = Pipeline([('CV',GridSearchCV(RandomForestClassifier(), parameters, cv = 5))])
    clf.fit(X_train, y_train)


    rf = Pipeline([('rf', RandomForestClassifier(n_estimators = 10, max_depth = 20).fit(X_train, y_train))])
    rf_scores = cross_val_score(rf, X_train, y_train, cv=10, scoring="f1")

    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    pipe = make_pipeline(StandardScaler(), DecisionTreeClassifier(max_depth=30))
    pipe.fit(X_train, y_train)  # apply scaling on training data
    Pipeline(steps=[('standardscaler', StandardScaler()),
                    ('dt', DecisionTreeClassifier(max_depth=30))])


    df = pd.read_csv('C:\\Users\\Naga Sai\\Desktop\\yaswant-main\\project\\data\\encoded_playlist_songs.csv')
    prob_preds = pipe.predict_proba(df.drop(['favorite','track_id'], axis=1))
    threshold = 0.30 # define threshold here
    preds = [1 if prob_preds[i][1]> threshold else 0 for i in range(len(prob_preds))]
    df['prediction'] = preds


    title = st.text_input('Enter the  title Of Playlist', 'Proj')
    
    create_playlist(sp, username, title, 'This playlist was created using python!')
    # Function to check if the playlist was created successfully

    playlist_id = fetch_playlists(sp,username)['id'][0]

    list_track = df.loc[df['prediction']  == 1]['track_id']
    enrich_playlist(sp, username, playlist_id, list_track)



    


def rand_page():
    a = st.radio("Yes or No", ['Yes', 'No'], 1)
    if a=="Yes":
        rand_rec()


def main():
    
    spr_sidebar()
    st.header("Music Recommendation ")
    st.markdown(
        '*Music recommendation* is a online Robust Music Recommendation Engine where in you can finds the best songs that suits your taste.  ') 
    
    

    if st.session_state.app_mode == 'recommend':
        rec_page()

    
    if st.session_state.app_mode == 'recommend_rand':
        rand_page()

 

    
    if st.session_state.app_mode == 'home':
        home_page()



# Run main()
if __name__ == '__main__':
    main()


