import sys
import os
import spotipy
import spotipy.util as util
from spotipy.oauth2 import SpotifyClientCredentials
import csv
import numpy as np
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from keras import optimizers
from keras.metrics import categorical_accuracy
from keras.models import Sequential
from keras.layers import Dense, Activation
from pickle import dump
from pickle import load
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import KFold

os.chdir("C:\\Users\\12244\\yoonp\\independentCS\\spotipy")

os.environ['SPOTIPY_CLIENT_ID'] = '4e2ba88cc79247a8acc4160d1510764f'
os.environ['SPOTIPY_CLIENT_SECRET'] = '1698c2d495ed4d6fba3967974ced8423'

client_credentials_manager = SpotifyClientCredentials()
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

artists = ["BTS", "Stray Kids", "EXO", "Monsta X", "NCT 127", "SEVENTEEN", "BLACKPINK", "GOT7", "ATEEZ", "NCT Dream", "TWICE",
            "SHINee", "LOONA", "DAY6", "Mamamoo", "iKON", "The Boyz", "ITZY", "Dreamcatcher", "VIXX", "Sunmi", "Super Junior",
            "SuperM", "X1", "Chung Ha", "Tomorrow x Together",  "GFriend", "HyunA", "Girls' Generation", "B.A.P", "BIGBANG",
            "EXID", "IU", "A.C.E", "WINNER", "IZ ONE", "f(x)", "Wanna One", "Everglow", "OneUs", "Cosmic Girls", 
            "NU EST", "TVXQ", "(G)I-DLE", "Momoland", "BtoB", "2NE1", "INFINITE", "OH MY GIRL", "S.E.S.", "Baby Vox", 
            "Girl’s Day", "I.O.I", "Apink", "Miss A", "Brown Eyed Girls", "Wonder Girls", "Sistar", "Lovelyz", "CLC", "AOA",
            "HELLOVENUS", "TAEYEON", "Ailee", "HEIZE", "Jeong Eun Ji", "Fromis_9", "Weki Meki", "Bolbbalgan4", "Gugudan"]

guys = ["BTS", "Stray Kids", "EXO", "Monsta X", "NCT 127", "SEVENTEEN", "GOT7", "ATEEZ", "NCT Dream", "SHINee", "DAY6", 
        "iKON", "The Boyz", "VIXX", "Super Junior", "SuperM", "X1", "Tomorrow x Together", "B.A.P", "BIGBANG", "A.C.E", 
        "WINNER", "Wanna One", "OneUs", "NU EST", "TVXQ", "BtoB", "INFINITE"]

girls = ['BLACKPINK', 'TWICE', 'LOONA', 'Mamamoo', 'ITZY', 'Dreamcatcher', 'Sunmi', 'Chung Ha', 'GFriend', 'HyunA',
         "Girls' Generation", 'EXID', 'IU', 'IZ ONE', 'f(x)', 'Everglow', 'Cosmic Girls', '(G)I-DLE', 'Momoland',
         '2NE1', 'OH MY GIRL', "S.E.S.", "Baby Vox", "Girl’s Day", "I.O.I", "Apink", "Miss A", "Brown Eyed Girls", 
         "Wonder Girls", "Sistar", "Lovelyz", "CLC", "AOA", "HELLOVENUS", "TAEYEON", "Ailee", "HEIZE", "Jeong Eun Ji",
         "Fromis_9", "Weki Meki", "Bolbbalgan4", "Gugudan"]
#from Fromis_9

path_to_artist_info = "C:\\Users\\12244\\yoonp\\independentCS\\spotipy\\artist_info\\"

def get_artist(name):
    results = sp.search(q='artist:' + name, type='artist')
    items = results['artists']['items']
    if len(items) > 0:
        return items[0]
    else:
        return None
    

def get_artist_features(name):
    artist = get_artist(name)
    results = sp.artist_albums(artist['id'])
    albums = results['items']
    album_ids = []
    for album in albums:
        album_ids.append(album['id'])
    
    # features: valence, acousticness, danceability,
    #           energy, instrumentalness, speechiness, & tempo, liveness
    features = []
    for album_id in album_ids:
        results = sp.album_tracks(album_id)
        tracks = results['items']

        for track in tracks:
            audioFeatures = sp.audio_features(track['id'])[0]
            pop = sp.track(track['id'])['popularity']
            features.append([track['name'], audioFeatures['duration_ms'], audioFeatures['key'], audioFeatures['mode'], 
                             audioFeatures['time_signature'], audioFeatures['acousticness'], audioFeatures['danceability'], 
                             audioFeatures['energy'], audioFeatures['instrumentalness'], audioFeatures['liveness'], 
                             audioFeatures['loudness'], audioFeatures['speechiness'], audioFeatures['valence'], 
                             audioFeatures['tempo'], pop])

    return features

def store_artist_features(name):
    with open(path_to_artist_info + name + '.csv', 'w', encoding="utf-8-sig", newline='') as csvfile:
        features = get_artist_features(name)
        writer = csv.writer(csvfile)
        writer.writerow(['Name', 'Duration (MS)', 'Key', 'Mode', 'Time Signature', 'Acousticness', 'Danceability', 
                        'Energy', 'Instrumentalness', 'Liveness', 'Loudness', 'Speechiness', 'Valence', 'Tempo', 'Popularity'])
        
        for row in features:
            writer.writerow(row)

artistsNG = ["Fromis_9", "Weki Meki", "Bolbbalgan4", "Gugudan"]

def get_every_artist_file():
    for artist in artists: #or artistsNG
        store_artist_features(artist)

def compile_all_artists():
    name = []
    duration = []
    key = []
    mode = []
    time_sig = []
    acousticness = []
    danceability = []
    energy = []
    instrumentalness = []
    liveness = []
    loudness = []
    speechiness = []
    valence = []
    tempo = []
    gender = []
    group = []
    popularity = []

    for artist in artists:
        if artist in guys:
            group_gender = 'M'
        else:
            group_gender = 'F'
        with open(path_to_artist_info + artist + ".csv", newline='', encoding='utf-8-sig') as csvfile:
            has_header = csv.Sniffer().has_header(csvfile.readline())
            csvfile.seek(0)  # Rewind.
            reader = csv.reader(csvfile)
            if has_header:
                next(reader)
            for row in reader:
                if float(row[1]) < 300000:
                    name.append(row[0])
                    group.append(artist)
                    gender.append(group_gender)
                    duration.append(float(row[1]))
                    key.append(float(row[2]))
                    mode.append(float(row[3]))
                    time_sig.append(float(row[4]))
                    acousticness.append(float(row[5]))
                    danceability.append(float(row[6]))
                    energy.append(float(row[7]))
                    instrumentalness.append(float(row[8]))
                    liveness.append(float(row[9]))
                    loudness.append(float(row[10]))
                    speechiness.append(float(row[11]))
                    valence.append(float(row[12]))
                    tempo.append(float(row[13]))
                    popularity.append(float(row[14]))


    with open("all_artist.csv", 'w', newline='', encoding='utf-8-sig') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Name of Song', 'Group', 'Gender', 'Duration (MS)', 'Key', 'Mode', 'Time Signature', 'Acousticness', 'Danceability', 
                            'Energy', 'Instrumentalness', 'Liveness', 'Loudness', 'Speechiness', 'Valence', 'Tempo', 'Popularity'])
            
        for i in range(len(name)):
            writer.writerow([name[i], group[i], gender[i], duration[i], key[i], mode[i], time_sig[i], acousticness[i], 
                            danceability[i], energy[i], instrumentalness[i], liveness[i], loudness[i], speechiness[i], valence[i], 
                            tempo[i], popularity[i]])

def get_data():
    excel_file = []
    with open("all_artist.csv", newline='', encoding='utf-8-sig') as csvfile:
        has_header = csv.Sniffer().has_header(csvfile.readline())
        csvfile.seek(0)  # Rewind.
        reader = csv.reader(csvfile)
        if has_header:
            next(reader)
        for row in reader:
            excel_file.append([row[0], row[1], float(row[3]), float(row[4]), float(row[5]), float(row[6]), float(row[7]), float(row[8]), 
                                   float(row[9]), float(row[10]), float(row[11]), float(row[12]), float(row[13]), 
                                   float(row[14]), float(row[15])])
    return excel_file

def get_data_by_gender():
    audio_features = np.empty((0,13))
    gender = np.empty((0,1))
    with open("all_artist.csv", newline='', encoding='utf-8-sig') as csvfile:
        has_header = csv.Sniffer().has_header(csvfile.readline())
        csvfile.seek(0)  # Rewind.
        reader = csv.reader(csvfile)
        if has_header:
            next(reader)
        for row in reader:
            gender = np.append(gender, row[2])
            audio_features = np.append(audio_features, [[float(row[3]), float(row[4]), float(row[5]), float(row[6]), float(row[7]), float(row[8]), 
                                   float(row[9]), float(row[10]), float(row[11]), float(row[12]), float(row[13]), 
                                   float(row[14]), float(row[15])]], axis=0)
    return audio_features, gender

def get_data_by_popularity():
    audio_features = np.empty((0,13))
    pop = np.empty((0,1))
    with open("all_artist.csv", newline='', encoding='utf-8-sig') as csvfile:
        has_header = csv.Sniffer().has_header(csvfile.readline())
        csvfile.seek(0)  # Rewind.
        reader = csv.reader(csvfile)
        if has_header:
            next(reader)
        for row in reader:
            #print(row[1])
            pop = np.append(pop, row[16])
            audio_features = np.append(audio_features, [[float(row[3]), float(row[4]), float(row[5]), float(row[6]), float(row[7]), float(row[8]), 
                                   float(row[9]), float(row[10]), float(row[11]), float(row[12]), float(row[13]), 
                                   float(row[14]), float(row[15])]], axis=0)
    return audio_features, pop

#dimensions of dataset = 13
x1, y1 = get_data_by_gender()
encoder = LabelEncoder()
encoder.fit(y1)
encoded_y1 = encoder.transform(y1)
x2, y2 = get_data_by_popularity()

artists = get_data()

def sounds_closest_to(f):
    least_squares = []
    for artist in artists:
        ls = 0
        for i in range(len(f)):
            ls += (f[i]-artist[i+2])**2
        least_squares.append(ls)
    indexOfMin = least_squares.index(min(least_squares))
    text = artists[indexOfMin][0] + " - " + artists[indexOfMin][1]
    return text

#gender
# baseline model
def create_baseline_gender():
    # create model
    model = Sequential()
    model.add(Dense(60, input_dim=13, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def create_baseline_regression():
    model = Sequential()
    model.add(Dense(13, input_dim=13, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

def start_nn_gender():
    estimators = []
    estimators.append(('standardize', StandardScaler()))
    model = KerasClassifier(build_fn=create_baseline_gender, epochs=100, batch_size=5, verbose=0)
    estimators.append(('mlp', model))
    pipeline = Pipeline(estimators)
    return pipeline

def start_nn_pop():
    estimators = []
    estimators.append(('standardize', StandardScaler()))
    estimators.append(('mlp', KerasRegressor(build_fn=create_baseline_regression, epochs=50, batch_size=5, verbose=0)))
    pipeline = Pipeline(estimators)
    return pipeline

def binary_baseline_test():
    pipeline = start_nn_gender()
    kfold = StratifiedKFold(n_splits=10, shuffle=True)
    results = cross_val_score(pipeline, x1, encoded_y1, cv=kfold)
    print("Standardized: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
    #BASELINE

def regression_baseline_test():
    pipeline = start_nn_pop()
    kfold = KFold(n_splits=10)
    results = cross_val_score(pipeline, x2, y2, cv=kfold)
    print("Standardized: %.2f (%.2f) MSE" % (results.mean(), results.std()))

def train_gender():
    #now fit training data
    pipeline_gender = start_nn_gender()
    fitted = pipeline_gender.fit(x1, encoded_y1)
    dump(pipeline_gender, open('pipeline_gender.pkl', 'wb'))

def train_pop():
    pipeline_pop = start_nn_pop()
    fitted = pipeline_pop.fit(x2, y2)
    dump(pipeline_pop, open('pipeline_pop.pkl', 'wb'))

def predict(f, model):
    predicted = model.predict([f])
    return predicted

def unpickle_gender():
    loaded_model_gender = load(open('pipeline_gender.pkl', 'rb'))
    return loaded_model_gender

def unpickle_pop():
    loaded_model_gender = load(open('pipeline_pop.pkl', 'rb'))
    return loaded_model_gender

#train_gender()
#train_pop()


