
# DISCLAIMER THIS DOES NOT WORKING AS OF NOW

from flask import Flask, render_template, request, jsonify
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFaceHub
from langchain.docstore import InMemoryDocstore
from langchain.indexes import SimpleIndex
import tempfile
import PyPDF2
import docx

# Load environment variables from a .env file
load_dotenv()

app = Flask(__name__)

# Global variable to store the vectorstore
vectorstore = None

# Hugging Face Model details
HF_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
EMBEDDING_MODEL_NAME = "sentence-transformers/all-mpnet-base-v1"
QA_MODEL_NAME = "google/flan-t5-base"

# Initialize SentenceTransformer model
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

# Initialize LangChain LLM and retriever
def initialize_model_and_retriever():
    # Hugging Face Hub LLM for QA
    llm = HuggingFaceHub(repo_id=QA_MODEL_NAME, model_kwargs={"temperature": 0.0}, huggingfacehub_api_token=HF_API_KEY)
    
    # Initialize an FAISS vectorstore for retrieval
    global vectorstore
    index = SimpleIndex()
    docstore = InMemoryDocstore()
    vectorstore = FAISS(embedding_model, index, docstore)
    
    # Return LLM and retriever (vectorstore)
    return llm, vectorstore.as_retriever()

# Initialize the model and retriever at the start of the application
llm, retriever = initialize_model_and_retriever()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['file']
    
    # Extract text from PDF or DOC and load it into LangChain's vectorstore
    global vectorstore
    
    if file.filename.endswith('.pdf'):
        extracted_text = extract_text_from_pdf(file)
    elif file.filename.endswith('.doc') or file.filename.endswith('.docx'):
        extracted_text = extract_text_from_doc(file)

    # Add document to the vectorstore for retrieval
    vectorstore.add_texts([extracted_text])

    return jsonify({'message': 'File uploaded successfully, you can now ask questions.'})

@app.route('/ask_question', methods=['POST'])
def ask_question():
    global vectorstore
    question = request.json.get('question')
    print(f"Received question: {question}")

    if not vectorstore:
        return jsonify({'error': 'No document uploaded. Please upload a file first.'})

    try:
        # Create a RetrievalQA chain
        qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
        
        # Run the query through the QA chain
        response = qa_chain.run(question)
        print(f"QA Chain response: {response}")
        
        return jsonify({'response': response})
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)})

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