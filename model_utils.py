import joblib
import os
from pathlib import Path

# Ensure models directory exists
MODELS_DIR = Path(__file__).parent / "models"
os.makedirs(MODELS_DIR, exist_ok=True)

# Model file path
MODEL_FILENAME = "diabetes_model.pkl"
BUNDLE_PATH = MODELS_DIR / MODEL_FILENAME

def save_model_bundle(bundle):
    """Save the model bundle to disk.
    
    Args:
        bundle (dict): Dictionary containing the model and metadata
    """
    if not isinstance(bundle, dict) or "model" not in bundle:
        raise ValueError("Invalid bundle format. Expected dict with 'model' key.")
    
    # Ensure the models directory exists
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    # Save the bundle
    joblib.dump(bundle, BUNDLE_PATH)
    print(f"Model saved to {BUNDLE_PATH}")

def load_model_bundle():
    """Load the model bundle from disk.
    
    Returns:
        dict: The loaded model bundle
    """
    if not os.path.exists(BUNDLE_PATH):
        raise FileNotFoundError(
            f"Model file not found at {BUNDLE_PATH}. "
            "Please run train.py first to train the model."
        )
    
    try:
        bundle = joblib.load(BUNDLE_PATH)
        if not isinstance(bundle, dict) or "model" not in bundle:
            raise ValueError("Invalid model bundle format.")
        return bundle
    except Exception as e:
        raise RuntimeError(f"Error loading model from {BUNDLE_PATH}: {str(e)}")
