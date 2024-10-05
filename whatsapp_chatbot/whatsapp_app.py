# Import necessary libraries
from flask import Flask, request
from twilio.rest import Client
from twilio.twiml.messaging_response import MessagingResponse
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import tempfile
import PyPDF2
import docx
import requests
from urllib.parse import urlparse

# Load environment variables from .env file
load_dotenv()

# Initialize Flask application
app = Flask(__name__)

# Twilio configuration
TWILIO_ACCOUNT_SID = os.getenv('TWILIO_ACCOUNT_SID')
TWILIO_AUTH_TOKEN = os.getenv('TWILIO_AUTH_TOKEN')
TWILIO_PHONE_NUMBER = os.getenv('TWILIO_PHONE_NUMBER')

# Initialize Twilio client with credentials
client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

# Global variables for document handling
embedding_model = None
index = None
document_texts = []
EMBEDDING_DIM = 768  # Dimension for 'all-mpnet-base-v1'
user_states = {}  # Track conversation state for each user

def initialize_model():
    """
    Initialize the SentenceTransformer model and FAISS index.
    This is called once when the application starts.
    """
    global embedding_model, index
    EMBEDDING_MODEL_NAME = "sentence-transformers/all-mpnet-base-v1"
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    index = faiss.IndexFlatL2(EMBEDDING_DIM)

# Initialize the model at startup
initialize_model()

@app.route('/webhook', methods=['POST'])
def webhook():
    """
    Handle incoming WhatsApp messages.
    This is the main function that Twilio calls when a message is received.
    """
    incoming_msg = request.values.get('Body', '').lower()
    sender = request.values.get('From', '')
    
    response = MessagingResponse()
    msg = response.message()
    
    user_state = user_states.get(sender, {'state': 'idle'})
    
    if incoming_msg == 'reset':
        user_states[sender] = {'state': 'idle'}
        msg.body("Bot has been reset. Send 'start' to begin.")
        return str(response)
    
    if user_state['state'] == 'idle':
        if incoming_msg == 'start':
            user_states[sender] = {'state': 'waiting_for_document'}
            msg.body("Please send a PDF or DOCX file, or a direct link to a PDF.")
        else:
            msg.body("Send 'start' to begin using the document QA bot.")
    
    elif user_state['state'] == 'waiting_for_document':
        media_url = request.values.get('MediaUrl0', '')
        if media_url:
            try:
                process_document(media_url)
                user_states[sender] = {'state': 'ready_for_questions'}
                msg.body("Document processed successfully. You can now ask questions about it!")
            except Exception as e:
                msg.body(f"Error processing document: {str(e)}")
        elif incoming_msg.startswith(('http://', 'https://')):
            try:
                if urlparse(incoming_msg).path.endswith('.pdf'):
                    process_document(incoming_msg)
                    user_states[sender] = {'state': 'ready_for_questions'}
                    msg.body("Document processed successfully. You can now ask questions about it!")
                else:
                    msg.body("Please send a direct link to a PDF file.")
            except Exception as e:
                msg.body(f"Error processing document link: {str(e)}")
        else:
            msg.body("Please send a PDF or DOCX file, or a direct link to a PDF.")
    
    elif user_state['state'] == 'ready_for_questions':
        if index.ntotal > 0:
            try:
                answer = get_answer(incoming_msg)
                msg.body(answer)
            except Exception as e:
                msg.body(f"Error generating answer: {str(e)}")
        else:
            msg.body("No document found. Please send a document first.")
    
    return str(response)

def process_document(url):
    """
    Download and process a document from a URL.
    Extracts text and adds it to the index.
    """
    response = requests.get(url)
    
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(response.content)
        temp_file_path = temp_file.name
    
    try:
        if url.lower().endswith('.pdf'):
            extracted_text = extract_text_from_pdf(temp_file_path)
        elif url.lower().endswith(('.doc', '.docx')):
            extracted_text = extract_text_from_doc(temp_file_path)
        else:
            raise ValueError("Unsupported file format")
        
        add_to_index(extracted_text)
    finally:
        os.unlink(temp_file_path)

def extract_text_from_pdf(file_path):
    """Extract text from a PDF file."""
    with open(file_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ''.join([page.extract_text() for page in pdf_reader.pages])
    return text

def extract_text_from_doc(file_path):
    """Extract text from a Word document."""
    doc = docx.Document(file_path)
    text = ' '.join([para.text for para in doc.paragraphs])
    return text

def add_to_index(text):
    """
    Process text and add it to the FAISS index.
    Also stores the text chunks for later retrieval.
    """
    global document_texts
    
    chunks = [text[i:i+512] for i in range(0, len(text), 512)]
    embeddings = embedding_model.encode(chunks)
    index.add(embeddings.astype('float32'))
    document_texts.extend(chunks)

def get_answer(question):
    """
    Find relevant text chunks for a given question and return them as an answer.
    """
    question_embedding = embedding_model.encode([question])[0].reshape(1, -1).astype('float32')
    k = 3  # number of similar contexts to retrieve
    distances, indices = index.search(question_embedding, k)
    
    relevant_contexts = [document_texts[i] for i in indices[0]]
    
    response = "Based on the document, here are the most relevant excerpts:\n\n"
    for i, context in enumerate(relevant_contexts, 1):
        response += f"{i}. {context.strip()}\n\n"
    
    return response[:1600]

if __name__ == "__main__":
    app.run(debug=True)