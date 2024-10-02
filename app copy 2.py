from flask import Flask, render_template, request, jsonify
from transformers import RagTokenizer, RagRetriever, RagSequenceForGeneration
import PyPDF2
import docx
import os
from dotenv import load_dotenv
from datasets import load_dataset

# Load environment variables from a .env file
load_dotenv()

app = Flask(__name__)
# Global variable to store extracted document text
extracted_text = ''

# Initialize the RAG components
model_name = "facebook/rag-token-base"

# Load the tokenizer and generator for RAG
tokenizer = RagTokenizer.from_pretrained(model_name)
generator = RagSequenceForGeneration.from_pretrained(model_name)

# Optional: Initialize the retriever
retriever = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    global extracted_text
    file = request.files['file']

    # Extract text from PDF or DOC
    if file.filename.endswith('.pdf'):
        extracted_text = extract_text_from_pdf(file)
    elif file.filename.endswith('.doc') or file.filename.endswith('.docx'):
        extracted_text = extract_text_from_doc(file)

    return jsonify({'message': 'File uploaded successfully, you can now ask questions.'})

@app.route('/ask_question', methods=['POST'])
def ask_question():
    global extracted_text
    question = request.json.get('question')
    print(f"Received question: {question}")

    if not extracted_text:
        return jsonify({'error': 'No document text available. Please upload a file first.'})

    try:
        # Preprocess the question
        question = preprocess_question(question)
        
        # Call the RAG pipeline with the extracted text and user question
        response = call_rag_pipeline(question)
        print(f"Pipeline response: {response}")
        return jsonify({'response': response})
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)})

def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ''
    for page_num in range(len(pdf_reader.pages)):
        text += pdf_reader.pages[page_num].extract_text()
    return text

def extract_text_from_doc(doc_file):
    doc = docx.Document(doc_file)
    text = ' '.join([para.text for para in doc.paragraphs])
    return text

def preprocess_question(question):
    # Remove unnecessary punctuation and whitespace
    question = question.strip()
    return question

def set_retriever(retriever_model_name):
    global retriever
    retriever = RagRetriever.from_pretrained(
        retriever_model_name, 
        index_name="exact", 
        use_dummy_dataset=True, 
        trust_remote_code=True
    )

def call_rag_pipeline(question):
    # Tokenize the question
    inputs = tokenizer(question, return_tensors="pt")
    
    if retriever:
        # Retrieve relevant passages
        retrieved_docs = retriever(inputs.input_ids, return_tensors="pt")
        context_input_ids = retrieved_docs.context_input_ids
        context_attention_mask = retrieved_docs.context_attention_mask
    else:
        # Use dummy context if no retriever is set
        context_input_ids = inputs.input_ids
        context_attention_mask = inputs.attention_mask
    
    # Prepare generation inputs
    generation_inputs = {
        "input_ids": inputs.input_ids,
        "attention_mask": inputs.attention_mask,
        "context_input_ids": context_input_ids,
        "context_attention_mask": context_attention_mask,
    }
    
    # Generate the answer
    generation_output = generator.generate(**generation_inputs)
    generated_text = tokenizer.batch_decode(generation_output, skip_special_tokens=True)[0]
    
    print(f"Generated text: {generated_text}")
    
    return generated_text

if __name__ == "__main__":
    app.run(debug=True)