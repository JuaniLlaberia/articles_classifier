import logging
import os
from flask import Blueprint, jsonify, request
from src.utils.validators.validate_articles import validate_articles
from src.models.xgb_classifier import XGBoostClassifier

classification_bp = Blueprint("classification", __name__)

@classification_bp.route("/classify", methods=["POST"])
def classify_articles():
    """
    Endpoint to classify articles category using our XGBoost model

    Request JSON:
        {
            "articles": [
                {"text": "..."},
                ...
            ],
            "enableProbabilities": true  # Optional
        }
    Returns:
        200: JSON with classification results, e.g.
            {
                "results": {
                    "predictions": [...],
                    "predictions_labels": [...],
                    "predictions_probabilities": [...]  # if enableProbabilities is true
                }
            }
        400: JSON with error message for invalid input
        500: JSON with error message for server/model errors
    """
    # JSON Validation
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    articlesJSON = request.get_json()

    if "articles" not in articlesJSON:
        return jsonify({"error": "Request body must have 'articles'"}), 400

    if len(articlesJSON["articles"]) == 0:
        return jsonify({"error": "You must provide at least 1 article to classify"}), 400

    isValid, errors = validate_articles(articles=articlesJSON["articles"])
    if not isValid:
        return jsonify({"error": "Some articles are missing the text", "details": errors}), 400

    # Running classification
    logging.info(f"Starting classification for {len(articlesJSON['articles'])}")

    # Get absolute path to the model file
    current_dir = os.path.dirname(os.path.abspath(__file__))

    model_path = os.path.join(current_dir, '../../models/xgboost_model_v1.pkl')
    model_path = os.path.normpath(model_path)
    encoder_path = os.path.join(current_dir, '../../models/label_encoder.pkl')
    encoder_path = os.path.normpath(encoder_path)

    # Run with or without probabilities
    enableProbabilities = bool(articlesJSON.get("enableProbabilities", False))

    xgb_pipeline = XGBoostClassifier(model_path=model_path, encoder_path=encoder_path)
    results, error = xgb_pipeline.run(data=articlesJSON["articles"], enableProbabilities=enableProbabilities)
    if error:
        return jsonify({"error": "Something went wrong when predicting categories for articles", "details": error}), 500

    return jsonify({"results": results}), 200