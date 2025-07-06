print("🔁 Script started...")

import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Sample movie data
data = {
    'movie_id': [1, 2, 3, 4, 5],
    'title': ['The Matrix', 'Inception', 'Interstellar', 'The Dark Knight', 'Memento'],
    'overview': [
        "A computer hacker learns about the true nature of his reality and his role in the war against its controllers.",
        "A thief who steals corporate secrets through the use of dream-sharing technology.",
        "A team of explorers travel through a wormhole in space in an attempt to ensure humanity's survival.",
        "When the menace known as the Joker emerges from his mysterious past, he causes chaos in Gotham.",
        "A man with short-term memory loss attempts to track down his wife's murderer."
    ]
}

df = pd.DataFrame(data)

# Vectorize the overview text
vectorizer = TfidfVectorizer(stop_words='english')
vectors = vectorizer.fit_transform(df['overview'])

# Calculate cosine similarity matrix
similarity = cosine_similarity(vectors)

# Save to pickle files
pickle.dump(df, open('movies.pkl', 'wb'))
pickle.dump(similarity, open('similarity.pkl', 'wb'))

print("✅ movies.pkl and similarity.pkl created successfully!")
