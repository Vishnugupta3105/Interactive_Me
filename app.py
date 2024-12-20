import streamlit as st
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import wikipediaapi
from gtts import gTTS
import time
import spacy
import json
import os
from google.oauth2 import service_account

# Ensure st.secrets contains the expected keys
if "google_credentials" in st.secrets:
    # Retrieve the JSON string from secrets
    google_credentials_json = st.secrets["google_credentials"]
    
    try:
        # Convert the JSON string to a dictionary
         google_credentials = json.loads(google_credentials_json)
       
        
        # Write the credentials to a temporary file
        with open("google_credentials.json", "w") as f:
            json.dump(google_credentials, f)
        
        # Set the environment variable to point to the temporary file
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = "google_credentials.json"
        
        # Validate necessary keys
        keys_needed = {"type", "project_id", "private_key_id", "private_key", 
                       "client_email", "client_id", "auth_uri", "token_uri", 
                       "auth_provider_x509_cert_url", "client_x509_cert_url"}
        missing = keys_needed.difference(google_credentials.keys())
        
        if missing:
            raise Exception(f"Missing keys: {missing}")
        
    except json.JSONDecodeError as e:
        st.error(f"Error decoding JSON: {e}")
else:
    st.error("Error: google_credentials key not found in secrets")

# Load credentials from Streamlit secrets
credentials_info = st.secrets["google_credentials"]
credentials = service_account.Credentials.from_service_account_info(json.loads(credentials_info))

# Example: Initialize a Google Cloud service
from google.cloud import storage  # Example service
client = storage.Client(credentials=credentials)

# Set custom page config for the Streamlit app
st.set_page_config(page_title="Sid_Bot: Your Interactive Storyteller", page_icon="📚", layout="wide", initial_sidebar_state="expanded")

# Add custom CSS
st.markdown("""
<style>
    .sidebar .sidebar-content {
        background-color: #f0f2f6;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
    }
    .stTextInput>div>div>input {
        border: 1px solid #ccc;
        padding: 5px;
        border-radius: 5px;
    }
    .image-container {
        text-align: center;
    }
    .image-container img {
        max-width: 200px;  /* Adjust the max-width as needed */
        height: auto;
        margin-bottom: 10px;
    }
    /* Custom font and underline for title */ 
    .custom-title { 
        font-family: 'Roboto', sans-serif; 
        font-size: 2em; /* Adjust the font size as needed */
        text-align: left;  
    }

    /* Hide Streamlit menu and footer */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

</style>
""", unsafe_allow_html=True)


# Define paths
BOOKS_PATH = "./Books"
MOVIES_PATH = "./Movies"
COVERS_PATH = "./covers"

# Get list of books and movies
books = [book.replace(".pdf", "") for book in os.listdir(BOOKS_PATH) if book.endswith(".pdf")]
movies = [movie.replace(".pdf", "") for movie in os.listdir(MOVIES_PATH) if movie.endswith(".pdf")]

#Ensure temp directory exists
if not os.path.exists('temp'):
    os.makedirs('temp')

# Initialize Streamlit App with custom title
st.markdown('<h1 class="custom-title">Sid_Bot: Your Interactive Chat Bot </h1>', unsafe_allow_html=True)

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "uploaded_file_path" not in st.session_state:
    st.session_state.uploaded_file_path = None    

# Function to load documents and create vectors
def vector_embedding(file_path):
    if 'vectors' not in st.session_state or st.session_state.file_path != file_path:
        st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        st.session_state.loader = PyPDFLoader(file_path)
        st.session_state.docs = st.session_state.loader.load()
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=500)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)
        st.session_state.file_path= file_path
# Load cover image
def load_cover_image(title):
    image_path = os.path.join(COVERS_PATH, f"{title}.jpg")
    if os.path.exists(image_path):
        return image_path
    return None

# Display category selection with images
st.sidebar.header("Select a Category")
category = st.sidebar.selectbox("Category", ["Book", "Movie"])

# Display titles with cover images
titles = books if category == "Book" else movies

st.sidebar.header(f"Select a {category}")
title = st.sidebar.selectbox("Title", titles)

# Upload user document
st.sidebar.header("Upload Your Document")
uploaded_file = st.sidebar.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file:
    with st.spinner("Processing your document..."):
        file_path = os.path.join("temp", uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        vector_embedding(file_path)
        st.session_state.uploaded_file_path = file_path
        st.session_state.current_title = None
        st.success(f"'{uploaded_file.name}' has been uploaded and processed!")
else:
    st.session_state.uploaded_file_path = None

# Display cover image 
if st.session_state.uploaded_file_path: 
    st.write(f"Uploaded document: {os.path.basename(st.session_state.uploaded_file_path)}") 
else: 
    cover_image = load_cover_image(title) 
    if cover_image: st.image(cover_image, width=200) # Set the width to 200px 
    else: 
        st.write(f"Cover image not found for {title}")


# Track selected title changes using session state
if 'current_title' not in st.session_state:
    st.session_state.current_title = None

# If the selected title has changed, load the new document
if title != st.session_state.current_title and not st.session_state.uploaded_file_path:
    st.session_state.current_title = title  # Update current title
    file_path = os.path.join(BOOKS_PATH if category == "Book" else MOVIES_PATH, f"{title}.pdf")
    
    if os.path.exists(file_path):
        start_time = time.time()  # Start the timer
        
        with st.spinner(f"Loading '{title}', please wait..."):
            vector_embedding(file_path)
        
        end_time = time.time()  # End the timer
        elapsed_time = end_time - start_time  # Calculate elapsed time
        
        st.success(f"'{title}' loaded successfully in {elapsed_time:.2f} seconds!")
    else:
        st.error(f"Document '{title}' not found. Please check the title and try again.")

#Function to cleanup uploaded file
def cleanup_uploaded_file():
    if st.session_state.uploaded_file_path and os.path.exists(st.session_state.uploaded_file_path):
        os.remove(st.session_state.uploaded_file_path)
        st.session_state.uploaded_file_path = None

# Ensure cleanup happens after processing
if st.session_state.uploaded_file_path:
    cleanup_uploaded_file()


# Function to query Wikipedia
def query_wikipedia(query):
    wiki_wiki = wikipediaapi.Wikipedia('en')
    page = wiki_wiki.page(query)
    if page.exists():
        return page.summary[:500]  # Returning first 500 characters of the summary
    else:
        return None
    
# Function to generate and play voice response using gTTS 
def generate_voice_response(text):
    tts = gTTS(text, lang='en', tld='co.in') 
    tts.save("response.mp3") 
    audio_file = open("response.mp3", "rb") 
    audio_bytes = audio_file.read() 
    return audio_bytes

# Load Spacy Model
nlp = spacy.load("en_core_web_sm") 
# Function to analyze text and extract entities 
def analyze_text(text): 
    doc = nlp(text)
    entities = [(entity.text, entity.label_) for entity in doc.ents] 
    return entities


# Display previous chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input field for user query
if prompt1 := st.chat_input("Ask me anything about the document!"):
    # Display user message in chat
    st.session_state.messages.append({"role": "user", "content": prompt1})
    with st.chat_message("user"):
        st.markdown(prompt1)

    # Process the user question
    if 'vectors' in st.session_state:
        try:
            llm = ChatGroq(groq_api_key=st.secrets["groq_api"], model_name="Llama3-8b-8192")

            prompt = ChatPromptTemplate.from_template(
                """
                Answer the question based on the provided context only.
                Please provide the most accurate response based on the question.
                <context>
                {context}
                <context>
                Question: {input}
                """
            )
            document_chain = create_stuff_documents_chain(llm, prompt)
            retriever = st.session_state.vectors.as_retriever()
            retrieval_chain = create_retrieval_chain(retriever, document_chain)

            response = retrieval_chain.invoke({'input': prompt1})
            answer = response['answer'].strip()

            # Display assistant response
            with st.chat_message("assistant"):
                st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})

            # Generate and play voice response
            audio_bytes = generate_voice_response(answer)
            st.audio(audio_bytes, format="audio/mp3")

        except Exception as e:
            error_message = f"An error occurred: {e}"
            with st.chat_message("assistant"):
                st.markdown(error_message)
            st.session_state.messages.append({"role": "assistant", "content": error_message})
