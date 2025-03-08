import requests
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the movie dataset
movies_df = pd.read_csv("TOP 100 IMDB MOVIES.csv")  # Ensure the CSV file is in the same directory

# Preprocessing: Handle missing values and select relevant columns
movies_df = movies_df[['Title', 'Genre', 'Year', 'IMDB Rating']].dropna()

# Convert column names to lowercase for easier access
movies_df.columns = ['title', 'genre', 'year', 'rating']

# Clean genre column by replacing separators
movies_df['cleaned_genre'] = movies_df['genre'].str.replace(" ", "").str.lower()

# Initialize CountVectorizer to transform genre text into numerical vectors
vectorizer = CountVectorizer(tokenizer=lambda x: x.split(","))
genre_matrix = vectorizer.fit_transform(movies_df['cleaned_genre'])

# Function to fetch movie genre from TMDb API
def get_movie_genre_tmdb(movie_title, api_key):
    search_url = f"https://api.themoviedb.org/3/search/movie?api_key={api_key}&query={movie_title}"
    response = requests.get(search_url).json()

    if response['results']:
        movie_id = response['results'][0]['id']
        details_url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={api_key}"
        movie_details = requests.get(details_url).json()
        genre_list = [genre['name'].lower().replace(" ", "") for genre in movie_details['genres']]
        return ", ".join(genre_list)
    else:
        return None

# Enter your TMDb API Key here
api_key = 'YOUR_API_KEY_HERE'  # Replace with your actual API key

# Function to recommend similar movies based on genre
def add_movie_from_tmdb_and_recommend(title, num_recommendations=5):
    # Fetch genre from TMDb API
    genre = get_movie_genre_tmdb(title, api_key)

    if genre:
        # Transform the new movie genre using the same vectorizer
        new_movie_vector = vectorizer.transform([genre])

        # Calculate cosine similarity with the existing genre matrix
        cosine_sim_new = cosine_similarity(new_movie_vector, genre_matrix)

        # Get similarity scores and sort them
        sim_scores = list(enumerate(cosine_sim_new[0]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        # Get the indices of the most similar movies
        sim_scores = sim_scores[:num_recommendations]
        movie_indices = [i[0] for i in sim_scores]

        # Display the recommended movies
        return movies_df[['title', 'genre', 'rating', 'year']].iloc[movie_indices]
    else:
        return "Movie not found in TMDb."

# Example: Get recommendations for a movie not in the dataset
new_movie_title = "Inception"
recommendations = add_movie_from_tmdb_and_recommend(new_movie_title)
print(recommendations)
