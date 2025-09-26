import os
import re
import pickle
import pandas as pd
import nltk
from nltk.stem import WordNetLemmatizer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

def setup_nltk():
    """Checks for NLTK 'wordnet' resource and downloads it if missing."""
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        print("Downloading NLTK 'wordnet' data...")
        nltk.download('wordnet')
    print("NLTK setup complete.")


def build_model():
    """The main function to build the model, now also saving the raw embeddings."""
    print("Starting ADVANCED model building process...")
    setup_nltk()

    # --- Step 1: Load and process dialogue data ---
    print("Step 1: Loading and processing dialogue data...")
    df = pd.read_json('script-bag-of-words.json')
    dialogue = {}
    for index, row in df.iterrows():
        for item in row['text']:
            dialogue[item['name']] = dialogue.get(item['name'], "") + " " + item['text']

    processed_df = pd.DataFrame()
    processed_df['character'] = dialogue.keys()
    processed_df['dialogue'] = dialogue.values()
    processed_df['num_words'] = processed_df['dialogue'].apply(lambda x: len(x.split()))
    processed_df = processed_df.sort_values('num_words', ascending=False).head(100)

    # --- Step 2: Lemmatization ---
    print("Step 2: Applying advanced text preprocessing (Lemmatization)...")
    lemmatizer = WordNetLemmatizer()
    processed_df['lemmatized_dialogue'] = processed_df['dialogue'].apply(
        lambda text: ' '.join([lemmatizer.lemmatize(word) for word in text.split()])
    )

    # --- Step 3: Build local image paths ---
    print("Step 3: Building local image paths...")
    final_characters_data = []
    placeholder_image = 'https://placehold.co/150x150/2d3748/e0e0e0?text=No+Image'
    
    for char in processed_df['character']:
        clean_name = re.sub(r'[\s-]+', '_', char).lower() + '.jpg'
        local_image_path = os.path.join('static', 'images', clean_name)
        
        if os.path.exists(local_image_path):
            image_url = os.path.join('images', clean_name).replace('\\', '/')
        else:
            image_url = placeholder_image
            
        final_characters_data.append({'character': char, 'image_url': image_url})

    final_df = pd.DataFrame(final_characters_data)
    
    # --- Step 4: Generate Sentence Embeddings ---
    print("\nStep 4: Generating sentence embeddings...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(processed_df['lemmatized_dialogue'].tolist(), show_progress_bar=True)

    # --- Step 5: Calculate and Save Similarity Matrix (for dropdown) ---
    print("Step 5: Calculating and saving cosine similarity matrix...")
    similarity_matrix = cosine_similarity(embeddings)
    pickle.dump(similarity_matrix, open('similarity.pkl', 'wb'))

    # --- NEW: Step 6: Save the Raw Embeddings (for text description) ---
    print("Step 6: Saving character embeddings...")
    pickle.dump(embeddings, open('embeddings.pkl', 'wb'))
    
    # --- Step 7: Save the final character data ---
    print("Step 7: Saving final character data...")
    pickle.dump(final_df, open('characters.pkl', 'wb'))
    
    print("\nModel building complete! 'characters.pkl', 'similarity.pkl', and 'embeddings.pkl' have been created.")


if __name__ == '__main__':
    build_model()

