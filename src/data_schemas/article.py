from pydantic import BaseModel

class Article(BaseModel):
    text: str
