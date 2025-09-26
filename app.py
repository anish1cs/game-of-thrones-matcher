from flask import Flask, render_template, request
import pickle
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

app = Flask(__name__)

# --- Load all models and data at startup ---
try:
    characters_df = pickle.load(open('characters.pkl', 'rb'))
    similarity_matrix = pickle.load(open('similarity.pkl', 'rb'))
    character_embeddings = pickle.load(open('embeddings.pkl', 'rb'))
    character_list = characters_df['character'].values
    # Load the SBERT model into memory
    sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
except FileNotFoundError:
    print("ERROR: Make sure all .pkl files exist. Run model.py first.")
    # Initialize empty variables to prevent crashes
    characters_df = pd.DataFrame(columns=['character', 'image_url'])
    character_list = []

# --- Routes ---

@app.route('/')
def index():
    """Renders the main page."""
    return render_template('index.html', characters=character_list)

@app.route('/match', methods=['POST'])
def match():
    """Handles the dropdown character matching."""
    selected_character_name = request.form.get('character')
    idx = characters_df[characters_df['character'] == selected_character_name].index[0]
    
    # Use the pre-calculated similarity matrix for speed
    similar_character_index = sorted(list(enumerate(similarity_matrix[idx])), reverse=True, key=lambda x: x[1])[1][0]

    selected_char_data = characters_df.iloc[idx].to_dict()
    matched_char_data = characters_df.iloc[similar_character_index].to_dict()

    return render_template('index.html',
                           characters=character_list,
                           selected_char=selected_char_data,
                           matched_char=matched_char_data)

# --- NEW: Route for handling text description ---
@app.route('/describe', methods=['POST'])
def describe():
    """Handles matching based on user's text description."""
    user_description = request.form.get('description')

    if not user_description or not user_description.strip():
        # If the description is empty, just reload the page
        return render_template('index.html', characters=character_list, error="Please enter a description.")

    # --- Real-time NLP Processing ---
    # 1. Create an embedding for the user's description
    user_embedding = sbert_model.encode([user_description])

    # 2. Calculate cosine similarity between the user's text and all characters
    similarities = cosine_similarity(user_embedding, character_embeddings)

    # 3. Find the index of the character with the highest similarity score
    best_match_index = np.argmax(similarities[0])
    
    matched_char_data = characters_df.iloc[best_match_index].to_dict()

    return render_template('index.html',
                           characters=character_list,
                           user_description=user_description, # Pass the description back to the page
                           matched_char=matched_char_data)


if __name__ == '__main__':
    app.run(debug=True)

