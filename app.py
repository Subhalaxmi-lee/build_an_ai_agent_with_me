from flask import Flask, render_template, request, jsonify
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import tempfile
import PyPDF2
import docx

# Load environment variables from a .env file
load_dotenv()

app = Flask(__name__)


def reset_index():
    global faiss_index, document_texts
    faiss_index = faiss.IndexFlatL2(EMBEDDING_DIM)
    document_texts = []

# Global variables
embedding_model = None
faiss_index = None  # Changed name to be more explicit
document_texts = []
EMBEDDING_DIM = 768  # Dimension for 'all-mpnet-base-v1'

# Initialize SentenceTransformer model
def initialize_model():
    """
    Initialize the SentenceTransformer model and FAISS index.
    This is called once when the application starts.
    """
    global embedding_model, faiss_index
    EMBEDDING_MODEL_NAME = "sentence-transformers/all-mpnet-base-v1"
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    
    # Initialize FAISS index
    faiss_index = faiss.IndexFlatL2(EMBEDDING_DIM)  # Changed variable name here too
# Initialize the model at the start
initialize_model()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['file']
    
    if file.filename.endswith('.pdf'):
        extracted_text = extract_text_from_pdf(file)
    elif file.filename.endswith('.doc') or file.filename.endswith('.docx'):
        extracted_text = extract_text_from_doc(file)
    else:
        return jsonify({'error': 'Unsupported file format'})
    
    # Add document to the index
    add_to_index(extracted_text)
    
    return jsonify({'message': 'File uploaded successfully, you can now ask questions.'})

@app.route('/ask_question', methods=['POST'])
def ask_question():
    question = request.json.get('question')
    
    try:
        """
        Find relevant text chunks for a given question and return them as an answer.
        """
        # Create embedding for the question
        question_embedding = embedding_model.encode([question])[0].reshape(1, -1).astype('float32')
        
        # Search similar contexts
        k = 3  # number of similar contexts to retrieve
        distances, indices = faiss_index.search(question_embedding, k)  # Use faiss_index here too
        
        # Get relevant contexts
        relevant_contexts = [document_texts[i] for i in indices[0]]
        
        # Combine contexts for the response
        response = "Based on the document, here are the most relevant excerpts:\n\n"
        for i, context in enumerate(relevant_contexts, 1):
            response += f"{i}. {context.strip()}\n\n"
        
        return jsonify({'response': response})
    except Exception as e:
        return jsonify({'error': str(e)})

def add_to_index(text):
    """
    Process text and add it to the FAISS index.
    Also stores the text chunks for later retrieval.
    """
    global document_texts, faiss_index  # Make sure to reference the global faiss_index
    
    # Split text into chunks
    chunks = [text[i:i+512] for i in range(0, len(text), 512)]
    
    if chunks:  # Only process if there are chunks
        # Get embeddings for chunks
        embeddings = embedding_model.encode(chunks)
        
        # Convert embeddings to float32 and correct shape
        embeddings_float32 = embeddings.astype('float32')
        
        # Add embeddings to FAISS index
        faiss_index.add(embeddings_float32)  # Use faiss_index instead of index
        
        # Store text chunks
        document_texts.extend(chunks)

def extract_text_from_pdf(file):
    """Extract text from a PDF using PyPDF2."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        file.save(temp_file.name)
        pdf_reader = PyPDF2.PdfReader(temp_file.name)
        text = ''
        for page_num in range(len(pdf_reader.pages)):
            text += pdf_reader.pages[page_num].extract_text()
    return text

def extract_text_from_doc(file):
    """Extract text from a DOCX file using python-docx."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as temp_file:
        file.save(temp_file.name)
        doc = docx.Document(temp_file.name)
        text = ' '.join([para.text for para in doc.paragraphs])
    return text

if __name__ == "__main__":
    app.run(debug=True)