import logging
import joblib
import numpy as np
from typing import Tuple
from xgboost import XGBClassifier
from pathlib import Path
from pandas import DataFrame
from sklearn.preprocessing import LabelEncoder
from src.data_processing.processing_pipeline import DataProcessingPipeline
from src.data_processing.steps.cleaner import Cleaner
from src.data_processing.steps.embedder import Embedder

class XGBoostClassifier:
    def __init__(self, model_path: str, encoder_path: str):
        """
        Initializes a new instances of the XGBoostClassifier class

        Args:
            model_path (str): Path where the xgb model is saved
        """
        self.model_path = Path(model_path)
        self.encoder_path = Path(encoder_path)
        self.model: XGBClassifier = None
        self.encoder: LabelEncoder = None
        self._load_model()
        self._load_encoder()

    def _load_model(self):
        """
        Load XGBoost model from provided path
        """
        try:
            self.model = joblib.load(self.model_path)
            logging.info(f"Model loaded successfully from {self.model_path}")
        except Exception as e:
            logging.error(f"Failed to load model using {self.model_path}: {str(e)}")
            raise

    def _load_encoder(self):
        """
        Load Label Encoder model from provided path
        """
        try:
            self.encoder = joblib.load(self.encoder_path)
            logging.info(f"Encoder loaded successfully from {self.encoder_path}")
        except Exception as e:
            logging.error(f"Failed to load encoder using {self.encoder_path}: {str(e)}")
            raise

    def _preprocess_data(self, data: list[dict[str, str]]) -> DataFrame:
        """
        Pre-Process articles using data processing pipeline

        Args:
            data (list[dict[str, str]]): List of articles each containing it's text
        Returns:
            DataFrame: Pandas dataframe containing the text and embeddings from all provided articles
        """
        MODEL = "jina/jina-embeddings-v2-base-es:latest"
        CHUNK_SIZE = 3000
        CHUNK_OVERLAP = 450
        THRESHOLD = 15000

        pipeline = DataProcessingPipeline(steps=[
            Cleaner(step_name="Cleanning"),
            Embedder(step_name="Embedding",
                     model_name=MODEL,
                     chunk_size=CHUNK_SIZE,
                     chunk_overlap=CHUNK_OVERLAP,
                     threshold=THRESHOLD
                    )
        ])

        df = DataFrame(data)
        processed_data = pipeline.run(df=df)

        return processed_data

    def _predict(self, embeddings: list[list[float]]) -> dict[str, any]:
        """
        Predict class using loaded model

        Args:
            embeddings (list[list[float]]): List of processed articles embeddings
        Returns:
            dict: {
                "predictions": np.ndarray,  # Predicted class indices
                "predictions_labels": np.ndarray,  # Predicted class labels (decoded)
            }
        """
        # Get predictions
        predictions = self.model.predict(embeddings)
        # Get labels from predictions
        predictions_labels = self.encoder.inverse_transform(predictions)

        return {
            "predictions": predictions.tolist(),
            "predictions_labels": predictions_labels.tolist()
        }

    def _predict_proba(self, embeddings: list[list[float]]) -> dict[str, any]:
        """
        Predict probabilities and max class using loaded model

        Args:
            embeddings (list[list[float]]): List of processed articles embeddings
        Returns:
            dict: {
                "predictions": np.ndarray,  # Predicted class indices
                "predictions_labels": np.ndarray,  # Predicted class labels (decoded)
                "predictions_probabilities": np.ndarray  # Predicted probabilities for each class
            }
        """
        # Get probabilities of each class
        predictions_proba = self.model.predict_proba(embeddings)
        # Get predictions from probabilities
        predictions = np.argmax(predictions_proba, axis=1)
        # Get labels from predictions
        predictions_labels = self.encoder.inverse_transform(predictions)

        return {
            "predictions": predictions.tolist(),
            "predictions_labels": predictions_labels.tolist(),
            "predictions_probabilities": predictions_proba.tolist()
        }

    def run(self, data: list[dict[str, str]], enableProbabilities: bool = False) -> Tuple[dict[str, any], str | None]:
        """
        Run XGBoostClassifier pipeline to predict articles classes/categories

        Args:
            data (list[dict[str, str]]): List of dictionaries containing the articles text
            enableProbabilities (bool): Boolean value to toggle probabilities
        Returns:
            Tuple[dict, str | None]: A tuple containing:
                - results (dict): Prediction results. If enableProbabilities is True, includes class indices, labels, and probabilities; otherwise, only class indices and labels
                - error (str | None): Error message if an error occurred, otherwise None
        """
        logging.info(f"Preprocessing {len(data)} articles")
        processed_data = self._preprocess_data(data=data)

        if "embedding" not in processed_data:
            error = "The 'processed_data' dataframe does not contain the 'embedding' column"
            logging.error(error)
            return {}, error

        embeddings = processed_data["embedding"]
        embeddings_to_predict = np.stack(embeddings)

        logging.info(f"Generating predictions using {'probabilities classes' if enableProbabilities else 'max class'}")
        if enableProbabilities:
            results = self._predict_proba(embeddings=embeddings_to_predict)
        else:
            results = self._predict(embeddings=embeddings_to_predict)

        return results, None