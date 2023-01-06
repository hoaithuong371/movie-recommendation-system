import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import TfidfVectorizer

st.set_page_config(
    page_title="Movie Recommendation",
    page_icon="🎥"
)


def load_data(data_name):
    data = pd.read_csv(data_name)

    data.isna().sum()

    data['certificate'].fillna('', inplace = True)

    data['meta_score'].fillna(0.0, inplace = True)

    data['gross'].apply(lambda x: float(str(x).lstrip('$').rstrip('M')) if x is not None else x)

    data['gross'].fillna(0.0, inplace = True)

    data['year'].apply(lambda x: str(x).replace('(', '').replace(')', ''))

    return data

def get_recommendation(title, cosine_sim, data):

    indices = pd.Series(data.index, index= data['title'])

    idx = indices[title]

    sim_scores = list(enumerate(cosine_sim[idx]))

    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    sim_scores = sim_scores[1:11]

    movie_indices = [i[0] for i in sim_scores]

    return data['title'].iloc[movie_indices]

def display(data1, data2):
    k = 0
    for indx, row in data1.iloc[data2.index].iterrows():
        k = k + 1
        with st.container():
            st.markdown('{}. {}'.format(k,row["title"]))
            c1, c2 = st.columns([1, 3])
            with c1:
                st.image(row["link_poster"], use_column_width = 'always')
            with c2:
                st.write(':scroll: Overview: {}'.format(row["overview"]))
                st.write(':trophy: Certifacte: {} | :clock1: Time: {} | :date: Year: {}'.format(row["certificate"], row["runtime"], row["year"]))
                st.write('Genre: {}'.format(row["genre"]))
                st.write(':busts_in_silhouette: Casts: {}, {}, {}, {}'.format(row["star_1"], row["star_2"], row["star_3"], row["star_4"]))
                st.write('Rating: {} :star:'.format(row["rating"]))
                st.markdown('Link IMDB: {}'.format(row["link_imdb"]))

def main():
    st.title("Movie Recommendatation system")

    st.sidebar.success("Select a page under!")

    menu = ["Ranking","Overview","Feature"]
    choice = st.sidebar.selectbox("Recommend by: ",menu)

    data = load_data("imdb_top_1000_movies.csv")

    if choice == "Ranking":

        st.subheader("TOP 10 MOVIES BY RANKING:")
        display(data, data['rating'][:10])
        
    elif choice == "Overview":

        st.subheader("Recommend by Overview")

        option = st.selectbox("Select Movie ", data['title'])

        if st.button("Recommend"):

            search_name = option

            if search_name is not None:

                try:
                    tfidf = TfidfVectorizer(stop_words='english')

                    tfidf_matrix = tfidf.fit_transform(data['overview'])

                    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

                    result = get_recommendation(search_name,cosine_sim,data)

                    display(data, result)

                except:
                    result = "Not Found"
                    st.warning(result)
        

    else:

        data['feature'] = data['director'] + ' ' + data['genre'].apply(lambda a: str(a).replace(',', ' ')) + ' ' + data['star_1'] + ' ' + data['star_2'] + ' ' + data['star_3'] + ' ' + data['star_4'] 

        st.subheader("Recommend by Feature")

        option = st.selectbox("Select Movie ", data['title'])

        if st.button("Recommend"):

            search_name = option

            if search_name is not None:

                try:
                    tfidf = TfidfVectorizer(stop_words='english')

                    tfidf_matrix = tfidf.fit_transform(data['feature'])

                    cosine_sim_2 = linear_kernel(tfidf_matrix, tfidf_matrix)

                    result_2 = get_recommendation(search_name,cosine_sim_2,data)

                    display(data, result_2)
                    
                except:
                    result_2 = "Not Found"
                    st.warning(result_2)
                    
            st.write(result_2)
        

if __name__ == '__main__':
    main()
