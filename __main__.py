import logging
from flask import Flask
from src.routes.classification import classification_bp

logging.basicConfig(filename='classifier.log', level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')

def create_app() -> Flask:
    """
    Create and add routes for Flask app
    """
    app = Flask(__name__)
    app.register_blueprint(classification_bp)

    return app

def main():
    """
    Main entry point for the Flask Articles Classifier API.
    Starts the web server to classify articles using trained XGBoost mdoel.
    """
    app = create_app()
    app.run(host="0.0.0.0", port=5000, debug=True)

if __name__ == "__main__":
    main()