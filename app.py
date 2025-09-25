from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Load the pre-processed data and similarity matrix
try:
    characters_df = pickle.load(open('characters.pkl', 'rb'))
    similarity = pickle.load(open('similarity.pkl', 'rb'))
    character_list = characters_df['character'].values
except FileNotFoundError:
    print("ERROR: Make sure 'characters.pkl' and 'similarity.pkl' files are in the root directory.")
    print("Please run the model.py script first to generate these files.")
    characters_df = pd.DataFrame(columns=['character', 'image_url']) # Create empty df to avoid crash
    similarity = []
    character_list = []


@app.route('/')
def index():
    """
    Renders the main page with the dropdown list of characters.
    """
    return render_template('index.html', characters=character_list)


@app.route('/match', methods=['POST'])
def match():
    """
    Handles the form submission, finds the most similar character,
    and re-renders the page with the results.
    """
    # Get the character selected by the user from the form
    selected_character_name = request.form.get('character')

    # Find the index of the selected character
    try:
        idx = characters_df[characters_df['character'] == selected_character_name].index[0]
    except IndexError:
        # Handle case where character is not found (shouldn't happen with dropdown)
        return render_template('index.html', characters=character_list, error="Character not found.")

    # Get the similarity scores for the selected character and find the most similar one
    # We sort by similarity and take the second one (index 1), as the most similar (index 0) is the character itself
    similar_character_index = sorted(list(enumerate(similarity[idx])), reverse=True, key=lambda x: x[1])[1][0]

    # Get the data for the selected character and the matched character and convert them to dictionaries
    selected_char_data = characters_df.iloc[idx].to_dict()
    matched_char_data = characters_df.iloc[similar_character_index].to_dict()

    # This is the crucial step: re-render the SAME template, but now pass in the results
    return render_template('index.html',
                           characters=character_list,
                           selected_char=selected_char_data,
                           matched_char=matched_char_data)


if __name__ == '__main__':
    app.run(debug=True)

