from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import os

from backend.model.speech_model import SpeechEmotionModel
from backend.utils.audio_utils import convert_to_wav

app = FastAPI(title="Speech Emotion Recognition")

app.mount("/static", StaticFiles(directory="frontend/static"), name="static")
templates = Jinja2Templates(directory="frontend/templates")

model = SpeechEmotionModel()


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


from typing import List

@app.post("/predict")
async def predict(files: List[UploadFile] = File(...)):
    results = []

    for file in files:
        wav_path = convert_to_wav(file)
        emotion, confidence = model.predict(wav_path)
        os.remove(wav_path)

        results.append({
            "file_name": file.filename,
            "emotion": emotion,
            "confidence": round(confidence, 2)
        })

    

    return {"results": results}
