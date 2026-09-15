import os
import joblib
import pandas as pd
from loguru import logger
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from src.utils import preprocess_data
from src.config import settings

class Train:

    def __init__(self):
        self.vectorizer = TfidfVectorizer(ngram_range=(1,2), min_df=2, max_df=0.95)
        self.model = LogisticRegression()
        self.X_test = None
        self.y_test = None
        self.is_fitted = False

    def _prepare_data(self, data):
        return preprocess_data(data)

    def _split_data(self, data):
        X_train, X_test, y_train, y_test = train_test_split(data["clean_email"], data["label"], train_size=0.80, random_state=42)
        return X_train, X_test, y_train, y_test

    def train(self, path=None):
        path = path or settings.training_data_path
        try:
            df = pd.read_csv(path)
            processed_data = self._prepare_data(df)
            X_train, X_test , y_train, y_test = self._split_data(processed_data)
            x_train_vectorized = self.vectorizer.fit_transform(X_train)
            self.model.fit(x_train_vectorized, y_train)
            self.X_test, self.y_test = X_test, y_test
            self.is_fitted = True

            return self.model, self.vectorizer
        except Exception as e:
            print(f"Error occured while training: {e}")
            raise

    def evaluate(self, X_test=None, y_test=None):
        if not self.is_fitted:
            raise RuntimeError("Model is not trained yet. Call train() first.")

        X_test = X_test if X_test is not None else self.X_test
        y_test = y_test if y_test is not None else self.y_test

        if X_test is None or y_test is None:
            raise ValueError("No test data available. Pass X_test/y_test or call train() first.")

        X_test_vec = self.vectorizer.transform(X_test)
        y_pred = self.model.predict(X_test_vec)

        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        print(f"Accuracy: {acc:.4f}")
        print(report)
        return {"accuracy": acc, "report": report}

    def save(self, path="artifacts.pkl"):
        if not self.is_fitted:
            raise RuntimeError("Model is not trained yet. Call train() first.")

        artifacts_path = os.path.join(settings.artifacts_path, path)
        artifact = {
            "model": self.model,
            "vectorizer": self.vectorizer
        }
        joblib.dump(artifact, artifacts_path)

    @classmethod
    def load(cls, path):
        instance = cls()
        artifacts = joblib.load(path)
        instance.model = artifacts.get("model")
        instance.vectorizer = artifacts.get("vectorizer")
        instance.is_fitted = True
        if instance.model is None or instance.vectorizer is None:
            raise ValueError(f"Artifact at {path} is missing 'model' or 'vectorizer'.")
        return instance

if __name__ == "__main__":
    trainer = Train()
    logger.info("Initializing training")
    try:
        trainer.train()
        logger.success("successfully trained classifier")

        logger.info("evaluating the model")
        metrics = trainer.evaluate()
        logger.info("mertics details", metrics)
        logger.success("successfully evaluated the model")

        logger.info("saving the model")
        trainer.save()
        logger.success("successfully saved the model at", settings.artifacts_path)
    except Exception as e:
        logger.error("error occured while training, evaluating or saving the model", e)






    