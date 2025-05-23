from fastapi import UploadFile
from research_agent import ResearchAssistantRAG
import os
# from scihub import scihub_pdf
from google_scolar.web_search import google_scholar_search,advanced_google_scholar_search


research_assistant = None

async def process_text_question(question: str, pdf_file : UploadFile = None, chat_history=None):
    global research_assistant
    if not research_assistant:
        create_object()
    # Process with RAG
    answer, updated_chat_history = research_assistant.rag_with_research(question)
    print('answer',answer)
    
    return {
        "answer": answer,
        "chat_history": updated_chat_history
    }

def create_object():
    global research_assistant
    research_assistant = ResearchAssistantRAG(advanced_google_scholar_search)
    return research_assistant
