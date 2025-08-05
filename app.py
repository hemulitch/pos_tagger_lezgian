# app.py
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from flair.models import SequenceTagger
from flair.data import Sentence

import uvicorn
import os

app = FastAPI()
templates = Jinja2Templates(directory="templates")

MODEL_PATH = os.getenv("MODEL_PATH", "models/final-model.pt")
tagger = SequenceTagger.load(MODEL_PATH)

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("form.html", {"request": request, "predictions": None})

@app.post("/", response_class=HTMLResponse)
async def predict(request: Request, sentence: str = Form(...)):
    flair_sentence = Sentence(sentence)
    tagger.predict(flair_sentence)
    pairs = [(token.text, token.get_tag(tagger.tag_type).value) for token in flair_sentence]
    return templates.TemplateResponse("form.html", {"request": request, "predictions": pairs, "input_sentence": sentence})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
