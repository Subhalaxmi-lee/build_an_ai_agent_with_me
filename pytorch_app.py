# DISCLAIMER THIS DOES NOT WORKING AS OF NOW
from flask import Flask, render_template, request, jsonify
from transformers import pipeline
import PyPDF2
import docx
import os
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

app = Flask(__name__)
# Global variable to store extracted document text
extracted_text = ''

# Initialize the question-answering pipeline with PyTorch
qa_pipeline = pipeline("question-answering", framework="pt")

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
        # Call the question-answering pipeline with the extracted text and user question
        response = call_qa_pipeline(extracted_text, question)
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

def call_qa_pipeline(context, question):
    # Split the context into chunks of 1000 characters
    chunk_size = 1000
    chunks = [context[i:i + chunk_size] for i in range(0, len(context), chunk_size)]
    
    answers = []
    for chunk in chunks:
        result = qa_pipeline(question=question, context=chunk)
        answers.append(result['answer'])
    
    # Aggregate answers (for simplicity, just join them here)
    aggregated_answer = ' '.join(answers)
    return aggregated_answer

if __name__ == "__main__":
    app.run(debug=True)