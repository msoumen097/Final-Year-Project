import pandas as pd
from collections import Counter
from statistics import mean
import numpy
import nltk
import requests
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import wordnet 
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from itertools import combinations
from flask import Flask,request, render_template
import pickle
# from Treatment import fetch_disease_details  # Assuming you have a `diseaseDetail` function
import os
RAPIDAPI_KEY = os.getenv("b8d220f445mshcc08d1ad3be5b70p1c08f5jsn1f48f640484e")  # Set in your environment


app = Flask(__name__)

# Load datasets and model
df_comb = pd.read_csv("Dataset/dis_sym_dataset_comb.csv")
df_norm = pd.read_csv("Dataset/dis_sym_dataset_norm.csv")

X = df_comb.iloc[:, 1:]
Y = df_comb.iloc[:, 0:1]

with open('lr_model.pkl', 'rb') as file:
    lr_model = pickle.load(file)
dataset_symptoms = list(df_comb.columns[1:])

# Preprocessing utilities
lemmatizer = WordNetLemmatizer()
splitter = RegexpTokenizer(r'\w+')

# Synonym finder
def synonyms(term):
    synonym_set = set()
    for syn in wordnet.synsets(term):
        synonym_set.update(syn.lemma_names())
    return synonym_set



@app.route('/')
def home():
    return render_template('home.html')  # Ensure home.html is the main page

@app.route('/index', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        user_input = request.form.get('symptoms', '').strip()
        
        if user_input:  # Ensure user_input is not empty
            user_symptoms = user_input.lower().split(',')

            # Process symptoms
            processed_user_symptoms = []
            for sym in user_symptoms:
                sym = sym.strip()
                sym = sym.replace('-', ' ').replace("'", '')
                sym = ' '.join([lemmatizer.lemmatize(word) for word in splitter.tokenize(sym)])
                processed_user_symptoms.append(sym)

            # Synonym expansion
            expanded_symptoms = set()
            for user_sym in processed_user_symptoms:
                user_sym_tokens = user_sym.split()
                for comb in range(1, len(user_sym_tokens) + 1):
                    for subset in combinations(user_sym_tokens, comb):
                        expanded_symptoms.update(synonyms(' '.join(subset)))
                expanded_symptoms.add(user_sym)

            # Match symptoms
            found_symptoms = set()
            for data_sym in dataset_symptoms:
                data_sym_tokens = data_sym.split()
                for user_sym in expanded_symptoms:
                    match_count = sum(1 for token in data_sym_tokens if token in user_sym)
                    if match_count / len(data_sym_tokens) > 0.5:
                        found_symptoms.add(data_sym)

            return render_template('results.html', found_symptoms=list(found_symptoms), stage="cooccur")

        # If user_input is empty, render the index page with an error message
        return render_template('index.html', error="Please enter valid symptoms.")

    # Handle GET request (render index.html)
    return render_template('index.html')


@app.route('/refine', methods=['POST'])
def refine():
    selected_symptoms = request.form.getlist('selected_symptoms')

    # Find co-occurring symptoms
    dis_list = set()
    final_symptoms = list(selected_symptoms)
    counter_list = []

    for symp in selected_symptoms:
        dis_list.update(set(df_norm[df_norm[symp] == 1]['label_dis']))

    for dis in dis_list:
        row = df_norm.loc[df_norm['label_dis'] == dis].values.tolist()
        row[0].pop(0)  # Remove the label_dis column
        for idx, val in enumerate(row[0]):
            if val != 0 and dataset_symptoms[idx] not in final_symptoms:
                counter_list.append(dataset_symptoms[idx])

    # Count co-occurring symptoms
    dict_symp = dict(Counter(counter_list))
    dict_symp_tup = sorted(dict_symp.items(), key=lambda x: x[1], reverse=True)

    return render_template('results.html', found_symptoms=[t[0] for t in dict_symp_tup], stage="final", final_symptoms=final_symptoms)

@app.route('/predict', methods=['POST'])
def predict():
    selected_symptoms = request.form.getlist('selected_symptoms')

    # Prepare input for model
    sample_x = [0] * len(dataset_symptoms)
    for sym in selected_symptoms:
        sample_x[dataset_symptoms.index(sym)] = 1

    predictions = lr_model.predict_proba([sample_x])[0]
    diseases = list(Y['label_dis'].unique())
    topk_indices = predictions.argsort()[-4:][::-1]
    results = [(diseases[idx], predictions[idx] * 100) for idx in topk_indices]

    return render_template('results.html', predictions=results, stage="predict")

def fetch_disease_details(disease_name):
    """
    Fetches details about a disease from the Healthgraphic API.

    Args:
        disease_name (str): Name of the disease to fetch details for.

    Returns:
        dict: A dictionary containing the API response or error details.
    """
    # URL encode the disease_name to handle special characters
    encoded_name = urllib.parse.quote(disease_name)
    url = f"https://{API_HOST}/api.healthgraphic.com/v1/conditions/{encoded_name}"

    headers = {
        "x-rapidapi-host": API_HOST,
        "x-rapidapi-key": API_KEY
    }

    try:
        # Make the GET request
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Raise HTTPError for 4xx and 5xx responses
        
        # Parse JSON response
        return response.json()
    except requests.exceptions.HTTPError as e:
        return {"error": f"HTTP Error occurred: {str(e)}"}
    except requests.exceptions.RequestException as e:
        return {"error": f"An error occurred: {str(e)}"}

@app.route('/details', methods=['GET', 'POST'])
def details():
    """
    Flask route to display details about a specific disease.

    Returns:
        str: Rendered HTML page with disease details or an error message.
    """
    disease_name = None
    details = None

    if request.method == 'POST':
        disease_name = request.form.get('disease_name')
        if disease_name:
            # Fetch disease details
            details = fetch_disease_details(disease_name)

    return render_template('details.html', disease_name=disease_name, details=details)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

