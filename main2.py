
import numpy as np
import pandas as pd


data = pd.read_csv('imdb_top_1000_movies.csv')


data.shape


data.head()

data.info()



data.isna().sum()


data['certificate'].fillna('', inplace = True)


data['meta_score'].fillna(0.0, inplace = True)


data['gross'].apply(lambda x: float(str(x).lstrip('$').rstrip('M')) if x is not None else x)
data['gross'].fillna(0.0, inplace = True)


data['year'].apply(lambda x: str(x).replace('(', '').replace(')', ''))


index = data['rating'][:10].index
data.iloc[index]




from sklearn.feature_extraction.text import TfidfVectorizer


tfidf = TfidfVectorizer(stop_words='english')


tfidf_matrix = tfidf.fit_transform(data['overview'])


tfidf_matrix.shape


from sklearn.metrics.pairwise import linear_kernel


cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)


indices = pd.Series(data.index, index= data['title'])

def get_recommendations(title, cosine_sim=cosine_sim):
    # lấy index của phim 
    idx = indices[title]

    # Get the pairwsie similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar movies
    sim_scores = sim_scores[1:11]

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar movies
    return data['title'].iloc[movie_indices]


# In[15]:


# Top 10 bộ phim có nội dung tương tự với 'The Dark Knight Rises'.
get_recommendations('The Dark Knight Rises')


# In[16]:


# giá trị cosine của top 10 bộ phim tương tự với The Dark Knight Rises
index = indices['The Dark Knight Rises']
scores = list(enumerate(cosine_sim[index]))
scores = sorted(scores, key=lambda x: x[1], reverse=True)
scores = scores[1:11]
scores


data['feature'] = data['director'] + ' ' + data['genre'].apply(lambda a: str(a).replace(',', ' ')) + ' ' + data['star_1'] + ' ' + data['star_2'] + ' ' + data['star_3'] + ' ' + data['star_4'] 



data['feature'].head(5)


from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(stop_words='english')

tfidf_matrix = tfidf.fit_transform(data['feature'])

tfidf_matrix.shape


from sklearn.metrics.pairwise import linear_kernel

cosine_sim_2 = linear_kernel(tfidf_matrix, tfidf_matrix)


indices = pd.Series(data.index, index= data['title'])

def get_recommendations(title, cosine_sim=cosine_sim_2):
   idx = indices[title]

   # Get the pairwsie similarity scores of all movies with that movie
   sim_scores = list(enumerate(cosine_sim[idx]))

   # Sort the movies based on the similarity scores
   sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

   # Get the scores of the 10 most similar movies
   sim_scores = sim_scores[1:11]

   # Get the movie indices
   movie_indices = [i[0] for i in sim_scores]

   # Return the top 10 most similar movies
   return data['title'].iloc[movie_indices]


get_recommendations('The Batman')


index = indices['The Dark Knight Rises']
scores = list(enumerate(cosine_sim[index]))
scores = sorted(scores, key=lambda x: x[1], reverse=True)
scores = scores[1:11]
scores
