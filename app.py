from flask import Flask, render_template, request, jsonify
import os
from dotenv import load_dotenv
import google.generativeai as genai
import re
import nltk

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Configure Gemini API
API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel('gemini-2.0-flash')

def preprocess_input(text):
    """
    Applies basic preprocessing:
    1. Lowercasing
    2. Punctuation removal
    3. Tokenization
    """
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        nltk.download('punkt_tab', quiet=True)
    tokens = nltk.word_tokenize(text)
    return tokens, text

def get_llm_response(prompt):
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error: {e}"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    data = request.get_json()
    question = data.get('question')
    
    if not question:
        return jsonify({'error': 'No question provided'}), 400

    # Preprocess (for display/logging purposes as per requirement)
    tokens, cleaned_text = preprocess_input(question)
    
    # Get Answer
    answer = get_llm_response(question)
    
    return jsonify({
        'question': question,
        'processed_question': cleaned_text,
        'tokens': tokens,
        'answer': answer
    })

if __name__ == '__main__':
    app.run(debug=True)
