**AI PDF Chatbot with Voice **

An AI-powered PDF chatbot built with Streamlit, LangChain, FAISS, and Google Gemini that allows users to upload PDF documents, ask questions about them, and receive answers both as text and voice.

The application extracts text from PDFs, creates embeddings, stores them in a FAISS vector database, and uses Gemini AI to answer questions based on the document content.

**Features**

Upload multiple PDF documents
Extract text using pdfplumber
Smart text chunking using LangChain
Semantic search using FAISS vector database
AI-powered answers using Google Gemini
Voice output using gTTS
Multiple English voice options
Download AI-generated answer audio

**Tech Stack**
Python
Streamlit
LangChain
Google Gemini AI
FAISS Vector Database
pdfplumber
gTTS (Google Text-to-Speech)
dotenv

**Project Structure**
AI-PDF-Chatbot
│
├── app.py                # Main Streamlit application
├── .env                  # API key configuration
├── requirements.txt      # Project dependencies
├── faiss_index/          # Generated vector store
└── README.md             # Project documentation

**Installation**

Clone the Repository
git clone https://github.com/yourusername/AI-PDF-Chatbot.git
cd AI-PDF-Chatbot
Create Virtual Environment
python -m venv venv

Activate it:

Windows

venv\Scripts\activate

Mac/Linux

source venv/bin/activate
**Install Dependencies**
pip install -r requirements.txt
🔑 Setup Environment Variables

Create a .env file in the project folder.

GOOGLE_API_KEY=your_google_gemini_api_key

Get your API key from Google Gemini AI Studio.

**Run the Application**
streamlit run app.py

The app will open in your browser:

http://localhost:8501
