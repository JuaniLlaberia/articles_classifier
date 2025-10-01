from typing import List
from src.data_schemas.article import Article

def validate_articles(articles: List[Article]):
    errors = []

    for idx, article in enumerate(articles):
        isMissing = "text" not in article
        if isMissing:
            errors.append({"index": idx, "error": "Missing text property"})

    if errors:
        return False, errors

    return True, None
