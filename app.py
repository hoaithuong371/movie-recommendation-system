import streamlit as st
import streamlit.components.v1 as stc

import pandas as pd
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import TfidfVectorizer


def load_data(data_name):
    data = pd.read_csv(data_name)

    data.isna().sum()

    data['certificate'].fillna('', inplace = True)

    data['meta_score'].fillna(0.0, inplace = True)

    data['gross'].apply(lambda x: float(str(x).lstrip('$').rstrip('M')) if x is not None else x)

    data['gross'].fillna(0.0, inplace = True)

    data['year'].apply(lambda x: str(x).replace('(', '').replace(')', ''))

    return data

def get_recommendation(title,cosine_sim,data):

    indices = pd.Series(data.index, index= data['title'])

    idx = indices[title]

    sim_scores = list(enumerate(cosine_sim[idx]))

    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    sim_scores = sim_scores[1:11]

    movie_indices = [i[0] for i in sim_scores]

    return data['title'].iloc[movie_indices]

def search_term_if_not_found(term,df):
	result_df = df[df['title'].str.contains(term)]
	return result_df

RESULT_TEMP = """
<div style="width:90%;height:100%;margin:1px;padding:5px;position:relative;border-radius:5px;border-bottom-right-radius: 60px;
box-shadow:0 0 15px 5px #ccc; background-color: #a8f0c6;
  border-left: 5px solid #6c6c6c;">
<h4>{}</h4>
<p style="color:blue;"><span style="color:black;">📈Score::</span>{}</p>
<p style="color:blue;"><span style="color:black;">🔗</span><a href="{}",target="_blank">Link</a></p>
<p style="color:blue;"><span style="color:black;">💲Price:</span>{}</p>
<p style="color:blue;"><span style="color:black;">🧑‍🎓👨🏽‍🎓 Students:</span>{}</p>
</div>
"""

def main():
    st.title("Movie Recommendatation system")

    menu = ["Ranking","Overview","Feature"]
    choice = st.sidebar.selectbox("Recommend Movies by",menu)

    data = load_data("imdb_top_1000_movies.csv")

    if choice == "Ranking":

        st.subheader("TOP 10 MOVIES BY RANKING:")
        index = data['rating'][:10].index
        data.iloc[index]
    
    elif choice == "Overview":

        st.subheader("Recommend by Overview")
        search_name = st.text_input("Search: ")

        if st.button("Recommend"):

            if search_name is not None:

                try:

                    tfidf = TfidfVectorizer(stop_words='english')

                    tfidf_matrix = tfidf.fit_transform(data['overview'])

                    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

                    result = get_recommendation(search_name,cosine_sim,data)

                except:

                    result = "Not Found"
                    st.warning(result)
                    st.info("Suggested Options include")

                    result_data = search_term_if_not_found(search_name,data)
                    st.dataframe(result_data)
                
            st.write(result)
        

    else:

        data['feature'] = data['director'] + ' ' + data['genre'].apply(lambda a: str(a).replace(',', ' ')) + ' ' + data['star_1'] + ' ' + data['star_2'] + ' ' + data['star_3'] + ' ' + data['star_4'] 

        st.subheader("Recommend by Feature")

        search_name = st.text_input("Search: ")

        if st.button("Recommend"):

            if search_name is not None:

                try:

                    tfidf = TfidfVectorizer(stop_words='english')

                    tfidf_matrix = tfidf.fit_transform(data['feature'])

                    cosine_sim_2 = linear_kernel(tfidf_matrix, tfidf_matrix)

                    result_2 = get_recommendation(search_name,cosine_sim_2,data)

                except:

                    result_2 = "Not Found"
                    st.warning(result_2)
                    st.info("Suggested Options include")

                    result_data = search_term_if_not_found(search_name,data)
                    st.dataframe(result_data)
                
            st.write(result_2)

    


    
if __name__ == '__main__':
    main()