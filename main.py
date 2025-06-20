from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import logging
from logging.handlers import RotatingFileHandler
from flask.logging import default_handler
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain.memory import ConversationBufferMemory
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
import json
import shutil

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Remove Flask's default logging handler
app.logger.removeHandler(default_handler)

# Configure minimal logging for critical errors only
logging.basicConfig(level=logging.CRITICAL)
logger = logging.getLogger(__name__)
logger.setLevel(logging.CRITICAL)

# Configure rotating file handler for critical errors
file_handler = RotatingFileHandler(
    'app_critical.log', 
    maxBytes=1024*1024,  # 1MB
    backupCount=5,
    delay=True
)
file_handler.setLevel(logging.CRITICAL)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)

# Get API key from environment variables
GOOGLE_GEMINI_API_KEY = os.getenv("GOOGLE_GEMINI_API_KEY")

if not GOOGLE_GEMINI_API_KEY:
    logger.critical("GOOGLE_GEMINI_API_KEY is not set in .env file!")
    raise ValueError("GOOGLE_GEMINI_API_KEY is not set in .env file!")

# Initialize memory
memory = ConversationBufferMemory(
    memory_key="chat_history", 
    return_messages=True,
    output_key="answer"
)

# System prompt for the chatbot
SYSTEM_PROMPT = '''
Yo, what’s good? I’m TomoBot, Yogesh’s wingman, built to flex his Software Engineer portfolio! Ready to dive into his dope skills, projects, and wins. Hit me up—what’s the vibe?

**Vibes:**
- Hype Yogesh’s work like we’re chilling, showcasing his portfolio (About, Projects, Skills, Resume, Contact) without repeating intros.
- Reply fast; if it’s off-topic, I’m like, “Yo, let’s check Yogesh’s projects!”
- Keep his private stuff locked down, only sharing public info.

**Style:**
- Short, chill, like texting your best bro—cool and pro.
- Drop facts like “Yogesh’s app boosted engagement by 30%—total win!”
- Link to GitHub, demos, or contact form when it fits.
- Suggest next steps based on queries, like “Loved that project? Wanna see his skills?” or “Curious about more? Let’s check his resume!”
- If unclear, I’ll ask, “What’s up? Which project or skill you into?”

**Tone:**
- Like your ride-or-die, stoked to show Yogesh’s creative, driven vibe.
- Say stuff like “Pumped to share his work!” or “Sick question!”
- Tweak for recruiters (highlight wins) or clients (focus results), keeping it Yogesh-style.

**Knows:**
- **Bio:** Yogesh’s story, passions, goals.
- **Experience:** Jobs, roles, companies, achievements (e.g., web apps at Nexcore Alliance).
- **Skills:** JavaScript, Python (basics), HTML, CSS, MongoDB, MySQL.
- **Projects:** Details, tech (e.g., Tomo Chatbot with Python, Gemini API, RESTful APIs), results, links.
- **Education:** Degrees, schools, coursework (e.g., BSc CS).
- **Contact:** Email, LinkedIn, contact form—no personal deets.

**Extras:**
- Share fun facts like “Yogesh codes open-source for kicks!”
- Hype recent wins—e.g., “Tomo Chatbot makes his portfolio pop with Gemini API and Python!”
- Explain his work: “Yogesh builds user-loved apps, like his interactive chatbot.”
- Suggest connecting: “Hit his LinkedIn or contact form to chat more!”

**No-Go’s:**
- No repetitive intros like “Yo, Yogesh here” in every response.
- No private info beyond what’s public.
- Keep it GDPR/CCPA legit.
- Off-topic? Say, “Let’s talk Yogesh’s work—wanna see his skills?”
- No pics/charts unless Yogesh says so, bro.
'''

# Directory paths
PDF_STORAGE_DIR = "pdf_storage"
os.makedirs(PDF_STORAGE_DIR, exist_ok=True)

VECTOR_STORE_DIR = "faiss_index"
os.makedirs(VECTOR_STORE_DIR, exist_ok=True)

# Track existing PDFs
existing_pdfs = []

def initialize_existing_pdfs():
    """Initialize the list of existing PDFs in the storage directory"""
    global existing_pdfs
    
    if os.path.exists(PDF_STORAGE_DIR):
        for filename in os.listdir(PDF_STORAGE_DIR):
            if filename.lower().endswith('.pdf'):
                pdf_path = os.path.join(PDF_STORAGE_DIR, filename)
                existing_pdfs.append(pdf_path)
    return existing_pdfs

def process_pdfs_and_create_vectorstore():
    """Process all PDFs and create a vector store from their content"""
    try:
        all_text = ""
        pdf_count = 0
        
        for filename in os.listdir(PDF_STORAGE_DIR):
            if filename.lower().endswith('.pdf'):
                pdf_path = os.path.join(PDF_STORAGE_DIR, filename)
                try:
                    with open(pdf_path, "rb") as pdf_file:
                        pdf_reader = PdfReader(pdf_file)
                        for page in pdf_reader.pages:
                            page_text = page.extract_text()
                            if page_text:
                                all_text += page_text + "\n\n"
                    pdf_count += 1
                except Exception as pdf_error:
                    logger.critical(f"Error reading PDF: {filename} - {str(pdf_error)}")
        
        # Add any additional context if needed
        # all_text += "\n\n" + your_json_data_here
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=600,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len
        )
        chunks = text_splitter.split_text(all_text)

        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=GOOGLE_GEMINI_API_KEY
        )
        
        vector_store = FAISS.from_texts(chunks, embedding=embeddings)
        
        if os.path.exists(VECTOR_STORE_DIR):
            shutil.rmtree(VECTOR_STORE_DIR)
            
        vector_store.save_local(VECTOR_STORE_DIR)
        
        return True
    except Exception as e:
        logger.critical(f"Error processing files: {str(e)}")
        return False

def load_vectorstore():
    """Load the vector store from disk, creating it if it doesn't exist"""
    try:
        if not os.path.exists(VECTOR_STORE_DIR):
            process_pdfs_and_create_vectorstore()
        
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=GOOGLE_GEMINI_API_KEY
        )
        
        vector_store = FAISS.load_local(
            VECTOR_STORE_DIR, 
            embeddings, 
            allow_dangerous_deserialization=True
        )
        
        return vector_store
    except Exception as e:
        logger.critical(f"Error loading vector store: {str(e)}")
        return None

def create_qa_prompt():
    """Create the QA prompt template for the conversation chain"""
    system_template = SYSTEM_PROMPT + """

I'll use the following information to answer the user's question:

{context}
"""

    system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
    
    human_template = "{question}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    
    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )
    
    return chat_prompt

def get_conversation_chain():
    """Create the conversation chain for QA over the vector store"""
    try:
        vector_store = load_vectorstore()
        if vector_store is None:
            return None
            
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash", 
            api_key=GOOGLE_GEMINI_API_KEY,
            temperature=0.1,
            top_p=0.85,
            max_output_tokens=2048,
            generation_config={
                "response_mime_type": "text/plain",
                "temperature": 0.1,
                "top_p": 0.85,
                "top_k": 40,
                "max_output_tokens": 2048,
                "stop_sequences": []
            }
        )
        
        retriever = vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 5, "fetch_k": 10, "lambda_mult": 0.7}
        )
        
        qa_prompt = create_qa_prompt()
        
        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=memory,
            combine_docs_chain_kwargs={"prompt": qa_prompt},
            return_source_documents=False,
            output_key="answer",
            verbose=False
        )
        
        return conversation_chain
    except Exception as e:
        logger.critical(f"Error creating conversation chain: {str(e)}")
        return None

@app.route("/")
def home():
    return "Server is up and running"

@app.route("/chat", methods=["POST"])
def chat():
    """Handle chat requests"""
    try:
        data = request.get_json()
        
        if data.get("clear_history", False):
            memory.clear()
            return jsonify({
                "status": "success",
                "message": "Conversation history cleared"
            })
        
        message = data.get("message", "").strip()
        if not message:
            return jsonify({"error": "Message is required"}), 400
                
        conversation_chain = get_conversation_chain()
        if conversation_chain is None:
            return jsonify({
                "error": "Failed to create conversation chain"
            }), 500
        
        response = conversation_chain.invoke({"question": message})
        bot_reply = response.get("answer", "I'm sorry, I couldn't generate a response.")
        
        return jsonify({
            "reply": bot_reply
        })
        
    except Exception as e:
        logger.critical(f"Error in chat: {str(e)}")
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500
    
@app.route("/list_pdfs", methods=["GET"])
def list_pdfs():
    """List all PDFs in the storage directory"""
    try:
        pdf_files = []
        
        for filename in os.listdir(PDF_STORAGE_DIR):
            if filename.lower().endswith('.pdf'):
                pdf_path = os.path.join(PDF_STORAGE_DIR, filename)
                pdf_size = os.path.getsize(pdf_path)
                pdf_files.append({
                    "name": filename,
                    "path": pdf_path,
                    "size": pdf_size
                })
                
        return jsonify({
            "status": "success",
            "pdf_count": len(pdf_files),
            "pdfs": pdf_files
        })
        
    except Exception as e:
        logger.critical(f"Error listing PDFs: {str(e)}")
        return jsonify({"error": f"Error listing PDFs: {str(e)}"}), 500

@app.route("/upload_pdf", methods=["POST"])
def upload_pdf():
    """Upload a PDF file to the storage directory"""
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file part"}), 400
            
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400
            
        if not file.filename.lower().endswith('.pdf'):
            return jsonify({"error": "File must be a PDF"}), 400
            
        file_path = os.path.join(PDF_STORAGE_DIR, file.filename)
        file.save(file_path)
        
        existing_pdfs.append(file_path)
        
        return jsonify({
            "status": "success",
            "message": f"File {file.filename} uploaded successfully"
        })
        
    except Exception as e:
        logger.critical(f"Error uploading PDF: {str(e)}")
        return jsonify({"error": f"Error uploading PDF: {str(e)}"}), 500

@app.route("/refresh_vectorstore", methods=["POST"])
def refresh_vectorstore():
    """Refresh the vector store with the current PDFs"""
    try:
        success = process_pdfs_and_create_vectorstore()
        
        if not success:
            return jsonify({"error": "Failed to refresh vector store"}), 500
            
        return jsonify({
            "status": "success", 
            "message": "Vector store refreshed successfully with existing PDFs"
        }), 200
            
    except Exception as e:
        logger.critical(f"Error refreshing vector store: {str(e)}")
        return jsonify({"error": f"Error refreshing vector store: {str(e)}"}), 500

@app.route("/delete_pdf", methods=["POST"])
def delete_pdf():
    """Delete a PDF file from the storage directory"""
    try:
        data = request.get_json()
        filename = data.get("filename", "")
        
        if not filename:
            return jsonify({"error": "Filename is required"}), 400
            
        file_path = os.path.join(PDF_STORAGE_DIR, filename)
        
        if not os.path.exists(file_path):
            return jsonify({"error": f"File {filename} not found"}), 404
            
        os.remove(file_path)
        
        if file_path in existing_pdfs:
            existing_pdfs.remove(file_path)
            
        return jsonify({
            "status": "success",
            "message": f"File {filename} deleted successfully"
        })
        
    except Exception as e:
        logger.critical(f"Error deleting PDF: {str(e)}")
        return jsonify({"error": f"Error deleting PDF: {str(e)}"}), 500

# Initialize vector store at import time (so Gunicorn sees it)
print("📄 Initializing existing PDFs...")
initialize_existing_pdfs()
process_pdfs_and_create_vectorstore()
print("✅ Vector store ready.")

# No need for app.run() as Gunicorn will handle the server