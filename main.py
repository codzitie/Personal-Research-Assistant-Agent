from fastapi import FastAPI, File, UploadFile,Form
from pydantic import BaseModel
import uvicorn
# from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from services import process_text_question, process_audio_question

app = FastAPI()
# class QuestionInput(BaseModel):
#     question: str

@app.get("/", response_class=HTMLResponse)
async def root():
    with open("index.html") as f:
        return HTMLResponse(content=f.read())

@app.post("/ask_text")
async def ask_text(question_input: str = Form(...)):
    # response = await process_text_question(question_input, pdf_file)
    # print('resp',response)
    return await process_text_question(question_input)

@app.post("/ask_audio")
async def ask_audio(pdf_file: UploadFile = File(...), audio_file: UploadFile = File(...)):

    return await process_audio_question(pdf_file, audio_file)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

    