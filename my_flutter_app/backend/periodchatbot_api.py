from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import os# Add this import
from langchain_huggingface import HuggingFaceEmbeddings  
from langchain_community.vectorstores import FAISS  # type: ignore
from langchain.text_splitter import CharacterTextSplitter  # type: ignore
from langchain.docstore.document import Document  # type: ignore

app = FastAPI()

# Normalization dictionary
NORMALIZATION_DICT = {

    "hunchha": "huncha","hunxa": "huncha","huncha": "huncha","hun6a":"huncha",
    "bhanya": "bhaneko","bhaneko": "bhaneko", "paarcha":"parcha",
    "bhanako": "bhaneko", "vaneko": "bhaneko" ,"k":"k","ke":"k","mahina": "period",
    "maina": "period","aauchha": "aaucha", "vanya": "bhaneko",
    "auchha": "aaucha","aaucha": "aaucha","aauxa": "aaucha","auxa": "aaucha",
    "auncha":"aaucha","aau6a":"aaucha","matbal": "matlab",
    "matlab":"matlab","garne":"garne","garney": "garne","kina": "kina",
    "kna": "kina","jharcha": "jharcha","garccha":"garcha","garxa":"garcha","gar6a":"garcha",
    "jharchha": "jharcha","jharxa": "jharcha","lakshan": "lakshan",
    "lakxan": "lakshan","lakshyan": "lakshan",
    "laksan": "lakshan","milcha": "milcha","milchha": "milcha",
    "milxa": "milcha","snan": "nuhauna",
    "nuhauna": "nuhauna","nuhaune": "nuhauna", "nwaune": "nuhauna", "nwauna": "nuhauna",
    "katthi": "kati","kati": "kati","katee": "kati","kate": "kati","din": "din","deen": "din",
    "lagyo": "lagyo","layo": "lagyo",
    "dheelo": "dhilo","dhilo": "dhilo", "dhila": "dhilo", "lagcha": "lagcha",
    "lagchha": "lagcha","lagxa": "lagcha", "laagcha":"lagcha",
    "lagdincha": "lagcha","lagdinxa": "lagcha","lagdinchha": "lagcha",
    "mahilale": "mahila le", "malila le": "mahila le", "mahila le": "mahila le",
    "ktle": "mahila le", "kt le": "mahila le", "kti le": "mahila le",
    "maile": "maile", "mahile": "maile", "mailey": "maile","garni":"garne", "garne":"garne",
    "chakkar": "chakkar", "rigata": "chakkar", "ringata" : "chakkar",
    "mahawari": "period", "mahinawari": "period", "mahina wari": "period","para sareko":"period",
    "means":"period","chui bhako":"period","nachune bhako":"period","mahine":"period",
    "chuna namilne": "period","dard": "dukhne", "dukhne": "dukhne",
    "means":"period","chui bhako":"period",
    "dukhai":"dukhne","sakccha":"sakcha","sakxa":"sakcha",
    "sakkcha":"sakcha","sak6a":"sakcha",
    "garbhawati":"pregnant","garbawati":"pregnant","garbhavati":"pregnant","safha":"safa",
    "sapha":"safa","tanav":"tanab","rakt":"ragat","ranga":"rang",
    "upayogi": "useful", "yedi": "if", "yadi": "if", "kta": "Purush", "kta": "purush", 
    "garbhavastama": "preganacy ma", "pregnency": "pregnancy",
    "atti": "ekdamai", "ekdam": "ekdamai", "aushadi": "aaushadi", 
    "aushadhi": "aushadhi",  "period": "periods",
    "mahinama":"period ma", "sadharan": "normally", "hudaina": "hunna",
    "nai": "hunna", "hoina": "haina", "vane ko": "bhaneko",
    "dard": "dukhne", "dukhne": "pain", "piuna": "khana", "piuda": "khada",
    "prayog": "use", "milcha": "huncha","lako ho": "lageko ho",
    "lakoho": "lageko ho", "lagya ho": "lageko ho", "andaa": "anda",
    "sad":"emotional","dukhi":"emotional",
    "dadh": "dhad", "dhaad": "dhad","hello": "hi", "hu": "hi", "hey": "hi", 
    "namaskar": "namaste", "hiii": "hi","k xa": "namaste", "k cha": "namaste", "hii": "hi",

}

def normalize_text(text) -> str:
    """Normalize text by replacing variants with standard terms."""
    if not isinstance(text, str):  # Handle NaN or numeric values
        return ""  
    for variation, standard in NORMALIZATION_DICT.items():
        text = text.replace(variation, standard)
    return text.strip()

def load_dataset(csv_path: str):
    """Load dataset from CSV, handle errors, and normalize text columns."""
    if not os.path.exists(csv_path):
        raise HTTPException(status_code=404, detail=f"File not found: {csv_path}")
    
    try:
        # Load CSV and handle malformed lines
        data = pd.read_csv(csv_path, on_bad_lines='skip')
        print(f"Dataset loaded with {len(data)} rows.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading CSV: {str(e)}")

    # Ensure required columns exist
    required_columns = {'question_en', 'response_en', 'question_ne', 'response_ne'}
    if not required_columns.issubset(data.columns):
        raise HTTPException(status_code=400, detail="Dataset missing required columns.")
    
    # Normalize text and handle NaN values
    for col in required_columns:
        data[col] = data[col].fillna("").apply(normalize_text)
    
    if data.empty:
        raise HTTPException(status_code=404, detail="Dataset is empty.")
    
    return data

def create_vector_store(data):
    """Create a vector store from the dataset."""
    documents = []
    for _, row in data.iterrows():
        if row["question_en"].strip():  # Avoid empty questions
            documents.append(Document(page_content=row["question_en"], metadata={"response": row["response_en"]}))
        if row["question_ne"].strip():
            documents.append(Document(page_content=row["question_ne"], metadata={"response": row["response_ne"]}))

    if not documents:
        raise HTTPException(status_code=500, detail="No valid documents to create vector store.")

    # Split text into chunks
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_docs = []
    for doc in documents:
        chunks = text_splitter.split_text(doc.page_content)
        split_docs.extend([Document(page_content=chunk, metadata=doc.metadata) for chunk in chunks])

    # Initialize HuggingFace Embeddings and FAISS
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(split_docs, embeddings)
    return vector_store

# Load dataset & create vector store on startup
dataset_path = os.getenv("DATASET_PATH", "backend/training_data.csv")

try:
    print("Loading dataset...")
    data = load_dataset(dataset_path)
    print("Creating vector store...")
    vector_store = create_vector_store(data)
    print("Vector store created successfully.")
except HTTPException as e:
    raise HTTPException(status_code=e.status_code, detail=f"Initialization Error: {e.detail}")
except Exception as e:
    raise HTTPException(status_code=500, detail=f"Unexpected error during startup: {str(e)}")

# Request model for chatbot
class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
def chat(request: ChatRequest):
    """Process user message and return chatbot response."""
    try:
        query = normalize_text(request.message)
        results = vector_store.similarity_search(query, k=1)
        response = results[0].metadata["response"] if results else "I'm sorry, I don't have an answer for that."
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chatbot error: {str(e)}")
# uvicorn backend.periodchatbot_api:app --reload --host 0.0.0.0 --port 8000
# uvicorn backend.api:app --host 0.0.0.0 --port 8001 --reload