Game of Thrones Character Matcher
A web application that finds the most similar Game of Thrones character based on an analysis of their complete dialogues from the show. This project uses natural language processing (NLP) with sentence embeddings to understand the semantic meaning behind what characters say, providing more nuanced matches than simple word counts.

Features
Semantic Character Matching: Select a character and find who they are most similar to based on the meaning of their words.

Advanced NLP Model: Utilizes a pre-trained Sentence-BERT model to generate high-quality vector embeddings of character dialogues.

Thematic UI: A custom-styled user interface with a Game of Thrones theme, including thematic fonts and imagery.

Local Image Handling: All character and background images are stored and served locally from the application.

Built with Flask: A lightweight Python web framework serves the model and the user interface.

Technology Stack
Backend: Python, Flask
-- Frontend: HTML, CSS

ML / NLP: Sentence-Transformers (for SBERT), Scikit-learn (for cosine similarity), Pandas, NLTK (for lemmatization)

Setup and Installation
Follow these steps to run the project on your local machine.

1. Clone the Repository

git clone [https://github.com/anish1cs/game-of-thrones-matcher.git](https://github.com/anish1cs/game-of-thrones-matcher.git)
cd game-of-thrones-matcher

2. Create and Activate a Virtual Environment

On Windows:

python -m venv venv
.\venv\Scripts\activate

On macOS/Linux:

python3 -m venv venv
source venv/bin/activate

3. Install Dependencies

pip install -r requirements.txt

4. Build the Model (Important!)
You must run the model.py script first. This script processes all the text, generates the similarity model, and creates the necessary .pkl files.

python model.py

(The first run may take a few minutes as it downloads the Sentence-BERT model and NLTK data.)

5. Run the Web Application

python app.py

Open your web browser and navigate to http://127.0.0.1:5000 to see the application live.

How It Works
Data Extraction: Dialogues for each character are extracted from the script-bag-of-words.json file.

Text Preprocessing: The dialogues are cleaned and lemmatized using NLTK to reduce words to their root form (e.g., "ruling" becomes "rule").

Sentence Embeddings: The pre-trained all-MiniLM-L6-v2 model from the Sentence-Transformers library is used to convert each character's complete dialogue into a 384-dimensional vector. This vector represents the semantic meaning of the text.

Similarity Calculation: Cosine similarity is calculated between all character vectors to create a similarity matrix. A score of 1 means identical dialogue, and 0 means completely different.

Matching: When a user selects a character, the application looks up this pre-computed matrix to find the character with the highest similarity score.
