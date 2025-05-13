from langchain_community.embeddings import HuggingFaceEmbeddings
from groq import Groq

from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama


GROQ_API_KEY = "ur api key"
MODEL_PATH = "sentence-transformers/all-MiniLM-l6-v2"

client = Groq(api_key=GROQ_API_KEY)

model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}

embeddings = HuggingFaceEmbeddings(
    model_name=MODEL_PATH,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

llm = ChatGroq(temperature=0, groq_api_key=GROQ_API_KEY, model_name="llama3-70b-8192")