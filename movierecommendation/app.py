import streamlit as st
import pickle

# Load data
movies = pickle.load(open('movies.pkl', 'rb'))
similarity = pickle.load(open('similarity.pkl', 'rb'))

st.title("🎬 Movie Recommendation System")

# Movie selection dropdown
selected_movie = st.selectbox("Choose a movie to get recommendations", movies['title'].values)

# Recommendation function
def recommend(movie):
    index = movies[movies['title'] == movie].index[0]
    distances = list(enumerate(similarity[index]))
    sorted_movies = sorted(distances, reverse=True, key=lambda x: x[1])[1:6]
    
    recommended_movies = []
    for i in sorted_movies:
        recommended_movies.append(movies.iloc[i[0]].title)
    return recommended_movies

# Button
if st.button('Recommend'):
    st.subheader("You might also like:")
    recommendations = recommend(selected_movie)
    for movie in recommendations:
        st.write("✅", movie)
