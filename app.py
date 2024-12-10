from flask import Flask, render_template, request
import os
import zipfile
import numpy as np
import joblib
from sklearn.feature_extraction.text import CountVectorizer
from PyPDF2 import PdfReader
import docx2txt
import torch
from transformers import BertTokenizer, BertModel
import uuid

app = Flask(__name__)

# Load the tokenizer and BERT model for embeddings
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased').to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

# Load the MLP model
mlp_model = joblib.load('super_brand_mlp_model.joblib')

# Function to read content from PDF or DOCX files
def read_file(file_path, file_type='pdf'):
    if file_type == 'pdf':
        reader = PdfReader(file_path)
        text = " ".join(page.extract_text() for page in reader.pages if page.extract_text())
    elif file_type == 'doc':
        text = docx2txt.process(file_path)
    else:
        raise ValueError("Unsupported file type")
    return text

# Function to preprocess and tokenize documents into n-grams
def get_ngrams(text, n_range=(1, 3)):
    ngram_sets = {}
    for n in range(n_range[0], n_range[1] + 1):
        vectorizer = CountVectorizer(analyzer="word", ngram_range=(n, n))
        ngram_matrix = vectorizer.fit_transform([text])
        ngram_sets[n] = set(vectorizer.get_feature_names_out())
    return ngram_sets

# Function to calculate Jaccard Similarity between two sets of n-grams
def calculate_jaccard_similarity(text1, text2, n_range=(1, 3)):
    ngram_sets1 = get_ngrams(text1, n_range)
    ngram_sets2 = get_ngrams(text2, n_range)
    
    total_similarity = 0
    count = 0

    # For each n-gram size, calculate the Jaccard similarity and average them
    for n in range(n_range[0], n_range[1] + 1):
        intersection = ngram_sets1[n] & ngram_sets2[n]
        union = ngram_sets1[n] | ngram_sets2[n]
        sim = len(intersection) / len(union) if union else 0
        total_similarity += sim
        count += 1
    
    # Return the average similarity across all n-gram sizes
    return round((total_similarity / count) * 100, 2) if count != 0 else 0.0

# Function to detect exact copies and calculate similarity between document pairs
def detect_exact_copies(doc_texts, n_range=(1, 3)):
    similarities = {}
    doc_names = list(doc_texts.keys())
    
    for i, doc1 in enumerate(doc_names):
        for j, doc2 in enumerate(doc_names):
            if i < j:  # Compare unique pairs
                similarity = calculate_jaccard_similarity(doc_texts[doc1], doc_texts[doc2], n_range=n_range)
                similarities[(doc1, doc2)] = similarity
                
    return similarities

# Function to split text into chunks for MLP prediction
def split_text_into_chunks(text, max_length=512):
    tokens = text.split()  # Split by whitespace
    chunks = [" ".join(tokens[i:i+max_length]) for i in range(0, len(tokens), max_length)]
    return chunks

# Function to create BERT embeddings for each chunk
def create_bert_embeddings(texts):
    embeddings = []
    for text in texts:
        inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = bert_model(**inputs.to(bert_model.device))
        embeddings.append(outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy())
    return np.array(embeddings)

# Function to predict AI-generated percentage using MLP
def predict_ai_generated(text):
    chunks = split_text_into_chunks(text)
    probabilities = []

    for chunk in chunks:
        inputs = tokenizer(chunk, return_tensors='pt', truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = bert_model(**inputs.to(bert_model.device))
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
        chunk_prob = mlp_model.predict_proba([embedding])[0][1] 
        probabilities.append(chunk_prob)

    avg_probability = np.mean(probabilities) * 100
    return round(avg_probability, 2)

# Function to handle zip extraction with folder structure
def extract_zip_with_structure(zip_file, extract_to="uploaded_docs"):
    unique_folder = os.path.join(extract_to, uuid.uuid4().hex[:8])
    os.makedirs(unique_folder, exist_ok=True)

    extracted_paths = {}  # Map cleaned names to full paths

    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        for member in zip_ref.namelist():
            target_path = os.path.join(unique_folder, member)

            # Ensure parent directories exist
            os.makedirs(os.path.dirname(target_path), exist_ok=True)

            if not member.endswith('/'):  # Handle only files, skip folders
                with zip_ref.open(member) as source, open(target_path, "wb") as target:
                    target.write(source.read())
                
                # Clean the document name (e.g., remove folder paths and extensions)
                base_name = os.path.basename(member)
                cleaned_name = os.path.splitext(base_name)[0].replace("_", " ").strip()

                # Map cleaned name to the full path
                extracted_paths[cleaned_name] = target_path

    return extracted_paths

@app.route("/", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        file = request.files["file"]
        
        if not file:
            return "No file uploaded!", 400

        folder_path = "uploaded_docs"
        try:
            # Extract files and get cleaned names with paths
            extracted_files = extract_zip_with_structure(file, folder_path)
        except zipfile.BadZipFile:
            return "Error: The uploaded file is not a valid ZIP archive.", 400

        # Process documents
        document_texts = {}
        for cleaned_name, file_path in extracted_files.items():
            if os.path.isfile(file_path):
                doc_type = 'pdf' if file_path.endswith('.pdf') else 'doc'
                document_texts[cleaned_name] = read_file(file_path, file_type=doc_type)

        # Predict AI-generated scores and calculate similarity
        ai_scores = {doc: predict_ai_generated(text) for doc, text in document_texts.items()}
        similarity_scores = detect_exact_copies(document_texts, n_range=(1, 3))

        return render_template(
            "report.html",
            documents=document_texts.keys(),
            ai_scores=ai_scores,
            similarity_scores=similarity_scores
        )

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True, port=5001)
