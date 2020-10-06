# href : https://www.kaggle.com/CooperUnion/anime-recommendations-database

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import itertools
import collections
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import jaccard_similarity_score # Jaccard Similarity



animes = pd.read_csv('D:/Users/str_aml/Desktop/recommended systems project/anime engine/anime.csv') # load the data
animes['genre'] = animes['genre'].fillna('None') # filling 'empty' data
animes['genre'] = animes['genre'].apply(lambda x: x.split(', ')) 
# split genre into list of individual genre

print(animes['genre'].head(5))

genre_data = itertools.chain(*animes['genre'].values.tolist()) # flatten the list
genre_counter = collections.Counter(genre_data)
genres = pd.DataFrame.from_dict(genre_counter, orient='index').reset_index().rename(columns={'index':'genre', 0:'count'})
genres.sort_values('count', ascending=False, inplace=True)


# Plot genre
f, ax = plt.subplots(figsize=(8, 12))
sns.set_color_codes("pastel")
sns.set_style("white")
sns.barplot(x="count", y="genre", data=genres, color='b')
ax.set(ylabel='Genre',xlabel="Anime Count")



## binary encode
genre_map = {genre: idx for idx, genre in enumerate(genre_counter.keys())}
def extract_feature(genre):
    feature = np.zeros(len(genre_map.keys()), dtype=int)
    feature[[genre_map[idx] for idx in genre]] += 1
    return feature
    
anime_feature = pd.concat([animes['name'], animes['genre']], axis=1)
anime_feature['genre'] = anime_feature['genre'].apply(lambda x: extract_feature(x))
print(anime_feature.head(150))




test_data = anime_feature.take([1,2,10])
print(test_data.head(10))
for row in test_data.iterrows():
    print('Similar anime like {}:'.format(row[1]['name']))
    search = anime_feature.drop([row[0]]) # drop current anime
    search['result'] = search['genre'].apply(lambda x: jaccard_similarity_score(row[1]['genre'], x))
    search_result = search.sort_values('result', ascending=False)['name'].head(5)
    for res in search_result.values:
        print('\t{}'.format(res))
    print()
    
    
 # For this analysis I'm only interest in finding recommendations for the TV category

anime_tv = animes[animes['type']=='TV']
anime_tv.head()   


# Find all genres represented
%matplotlib inline

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import itertools
import collections
import operator

genres = set()


# List genres by count
genres_count = collections.defaultdict(int)
for entry in animes['genre']:
    if not type(entry) is str:
        continue
    seen_already = set()
    for genre in entry.split(", "):
        if genre in seen_already:
            continue
        seen_already.add(genre)
        genres_count[genre] += 1
sorted(genres_count.items(), key=operator.itemgetter(1), reverse=True)



fig = plt.figure(figsize=(20,20))
ax = plt.gca()
plt.title('All Animes Rating vs. Popularity By Genre')
plt.xlabel('Rating')
plt.ylabel('Popularity (People)')
num_colors = len(genres)
cm = plt.get_cmap('gist_rainbow')
ax.set_prop_cycle('color', [cm(1. * i / num_colors) for i in range(num_colors)])
ax.set_yscale('log')
# For each genre, plot data point if it falls in that category
for genre in genres:
    data_genre = data[data.genre.str.contains(genre) == True]
    ax.plot(data_genre["rating"], data_genre["members"], marker='o', linestyle='', ms=12, label=genre)
ax.legend(numpoints=1, loc='upper left');

