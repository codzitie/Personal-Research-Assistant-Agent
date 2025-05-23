from langchain_community.embeddings import HuggingFaceEmbeddings
from groq import Groq
from langchain_google_genai import ChatGoogleGenerativeAI   
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from langchain_deepseek import ChatDeepSeek
import os
import ollama

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Access them
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GOOGLE_API_KEY = os.getenv("GEMINI-API")
print('groq',GROQ_API_KEY,'\n','gemini',GOOGLE_API_KEY)
MODEL_PATH = "sentence-transformers/all-MiniLM-l6-v2"
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
client = Groq(api_key=GROQ_API_KEY)

model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}

embeddings = HuggingFaceEmbeddings(
    model_name=MODEL_PATH,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

# llm = ChatGroq(temperature=0, groq_api_key=GROQ_API_KEY, model_name="llama3-70b-8192")
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",  
    temperature=0.1,
    convert_system_message_to_human=True 
)
# response = ollama.chat(
#         model="deepseek-r1",
#         messages=[{"role": "user", "content": formatted_prompt}],
#     )
# llm = ChatOllama(model="llama3.2")