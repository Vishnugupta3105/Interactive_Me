import streamlit as st
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
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
import spacy
import time

# Set custom page config for the Streamlit app 
st.set_page_config( page_title="Sid_Bot: Your Interactive Storyteller", page_icon="📚", layout="wide", initial_sidebar_state="expanded" )


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
    .custom-title { font-family: 'Roboto', sans-serif; 
    font-size: 2em; /* Adjust the font size as needed */
    text-align: left;
            
    }
</style>
            
""", unsafe_allow_html=True)


# Load environment variables from .env file
load_dotenv()

# Explicitly set the GOOGLE_APPLICATION_CREDENTIALS environment variable
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'C:/Users/Hp/Desktop/Interaction/cashmitra-b04d8a88d121.json'
print("GOOGLE_APPLICATION_CREDENTIALS:", os.getenv('GOOGLE_APPLICATION_CREDENTIALS'))

# Define paths
BOOKS_PATH = "./Books"
MOVIES_PATH = "./Movies"
COVERS_PATH = "./covers"

# Get list of books and movies
books = [book.replace(".pdf", "") for book in os.listdir(BOOKS_PATH) if book.endswith(".pdf")]
movies = [movie.replace(".pdf", "") for movie in os.listdir(MOVIES_PATH) if movie.endswith(".pdf")]


# Initialize Streamlit App with custom title
st.markdown('<h1 class="custom-title">Sid_Bot: Your Interactive Storyteller</h1>', unsafe_allow_html=True)

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []


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
if category == "Book": 
    titles = books 
else: 
    titles = movies 
    
st.sidebar.header(f"Select a {category}") 
title = st.sidebar.selectbox("Title", titles) 

cover_image = load_cover_image(title)
if cover_image:
    st.image(cover_image, width=200)  # Set the width to 200px
else:
    st.write(f"Cover image not found for {title}")
  

# Function to load documents and create vectors (no change needed)
def vector_embedding(file_path):
    st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    st.session_state.loader = PyPDFLoader(file_path)
    st.session_state.docs = st.session_state.loader.load()
    st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=500)
    st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)
    st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)

# Track selected title changes using session state
if 'current_title' not in st.session_state:
    st.session_state.current_title = None

# If the selected title has changed, load the new document
if title != st.session_state.current_title:
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
            llm = ChatGroq(groq_api_key=os.getenv('GROQ_API_KEY'), model_name="Llama3-8b-8192")
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

