import streamlit as st
import os
import pdfplumber
import google.generativeai as genai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from gtts import gTTS
from io import BytesIO

# Load API key from .env
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

# Streamlit UI
st.title("📚 AI PDF Chatbot with Voice 🎤")

# Upload PDFs
uploaded_files = st.file_uploader("Upload PDF Files", accept_multiple_files=True, type=["pdf"])

def get_pdf_text(pdf_docs):
    """Extract text from PDFs using pdfplumber."""
    text = ""
    for pdf in pdf_docs:
        with pdfplumber.open(pdf) as pdf_reader:
            for page in pdf_reader.pages:
                text += page.extract_text() or ""  # Handle empty pages
    return text

def get_text_chunks(text):
    """Split text into smaller chunks for embeddings."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=100)
    return text_splitter.split_text(text)

def get_vector_store(text_chunks):
    """Generate FAISS vector store from text chunks."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embeddings)
    vector_store.save_local("faiss_index")
    return vector_store

def get_conversational_chain():
    """Create the question-answering chain with a custom prompt."""
    prompt_template = """
    Answer the question based on the provided context.
    If the answer is not in the context, say: 'Answer is not available in the context.'

    Context:\n {context}\n
    Question:\n {question}\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

def generate_speech(answer_text, lang="en", tld="com"):
    """Convert text to speech with gTTS and return audio file as BytesIO."""
    tts = gTTS(answer_text, lang=lang, tld=tld)  # Generate speech

    # Save the audio to a BytesIO object
    audio_bytes = BytesIO()
    tts.write_to_fp(audio_bytes)  # Correct way to store in memory
    audio_bytes.seek(0)  # Reset buffer position for reading

    return audio_bytes


if uploaded_files:
    raw_text = get_pdf_text(uploaded_files)
    text_chunks = get_text_chunks(raw_text)
    vector_store = get_vector_store(text_chunks)
    qa_chain = get_conversational_chain()

    # User input
    user_question = st.text_input("Ask a question about the PDFs:")
    
    # Voice selection dropdown
    voice_option = st.selectbox("Choose Voice:", ["US English", "British English", "Indian English", "Australian English"])
    voice_map = {
        "US English": ("en", "com"),
        "British English": ("en", "co.uk"),
        "Indian English": ("en", "co.in"),
        "Australian English": ("en", "com.au"),
    }
    
    if user_question:
        docs = vector_store.similarity_search(user_question, k=3)
        answer = qa_chain.run(input_documents=docs, question=user_question)

        # Display answer
        st.write("### 🤖 Answer:")
        st.success(answer)

        # Convert answer to speech
        selected_lang, selected_tld = voice_map[voice_option]
        audio_file = generate_speech(answer, lang=selected_lang, tld=selected_tld)

        # Play & Download audio
        st.audio(audio_file, format="audio/mp3")
        st.download_button(label="Download Answer Audio 🎧", data=audio_file, file_name="answer.mp3", mime="audio/mp3")
