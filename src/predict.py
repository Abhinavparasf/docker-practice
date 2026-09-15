from loguru import logger

from src.utils import process_text
from src.config import settings
from src.train import Train

_loader = Train.load(settings.artifacts_path / "artifacts.pkl")

def predict(text: str)-> str:
    if not text.strip():
        logger.warning("input is empty")
        return "empty text can't be processed"
    logger.info("loading vectorizer and model")
    vectorizer = _loader.vectorizer
    model = _loader.model

    input_text = process_text(text)
    vectorized_text = vectorizer.transform([input_text])
    logger.info("predicting the category")
    try:
        predicted_class = model.predict(vectorized_text)
        logger.success(f"predicted class : {predicted_class[0]}")
        return int(predicted_class[0])
    except Exception as e:
        logger.error("error occured while predicting the category")
        raise