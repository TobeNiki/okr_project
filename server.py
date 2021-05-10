import uvicorn
from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import List
import time
from TextSimlarity import TextSimilarity
from Morpheme import Morpheme

app = FastAPI()
morpheme = Morpheme()

class Data(BaseModel):
    corpus : List[str]
    morpheme_on : bool

@app.on_event("startup")
async def startup_event():
    print("__startup event__")
    
@app.on_event("shutdown")
def shutdown_event():
    print("__shutdown__")

@app.post('/text_simila') # methodとendpointの指定
async def text_simila(data : Data):
    if data.morpheme_on:
        text_corpus = morpheme.fit_transform(data.corpus)
    else:
        text_corpus = data.corpus
    textsimila = TextSimilarity()
    cos_ts = textsimila.fit_transform(text_corpus)
    return {
        "cos_similarity":cos_ts.tolist(), 
        "tfidf":  textsimila.get_tfidf().tolist(),
        "feature_name":textsimila.get_feature_name()
    }

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

