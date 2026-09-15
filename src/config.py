import os
from pathlib import Path
from pydantic import BaseModel

file_path = Path(__file__).resolve()

parent_dir = file_path.parent.parent

class Settings(BaseModel):
    training_data_path: str = parent_dir / "data" / "spam_or_not_spam.csv"
    artifacts_path: str = parent_dir / "models"

settings = Settings()

    
