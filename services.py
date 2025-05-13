from fastapi import UploadFile
from utils import save_upload_file_temp, remove_temp_file, transcribe_and_rag,ResearchAssistantRAG
import os
from google_scolar.web_search import google_scholar_search

research_assistant = None

async def process_text_question(question: str, pdf_file : UploadFile = None, chat_history=None):
    global research_assistant
    if pdf_file:
        temp_pdf_path = f"temp_{pdf_file.filename}"
        with open(temp_pdf_path, "wb") as f:
            f.write(await pdf_file.read())
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
    research_assistant = ResearchAssistantRAG(google_scholar_search)
    return research_assistant

async def process_audio_question(pdf_file: UploadFile, audio_file: UploadFile):
    pdf_path = await save_upload_file_temp(pdf_file)
    audio_path = await save_upload_file_temp(audio_file)
    result = transcribe_and_rag(pdf_path, audio_path)
    remove_temp_file(pdf_path)
    remove_temp_file(audio_path)
    return {"answer": result}