import time
import uvicorn
from loguru import logger
from fastapi import FastAPI
from fastapi import HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.models import RequestSchema, ResponseSchema
from src.predict import predict

MAPPING = {0: "Not a Spam", 1: "Spam"}

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict", response_model=ResponseSchema)
def classify_email(payload: RequestSchema):
    logger.info("Initializing application")
    if not payload.text.strip():
        raise HTTPException(status_code=422, detail="Text cannot be empty")
    try:
        start_time = time.time()
        predicted_class = int(predict(payload.text))
        end_time = time.time()
        logger.info(f"Time Taken to classify : {end_time - start_time}")
    except Exception as e:
        raise HTTPException(status_code=500, detail="Prediction failed") from e
    return ResponseSchema(category=MAPPING[predicted_class])


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)