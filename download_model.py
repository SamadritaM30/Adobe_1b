# download_model.py

import os
from sentence_transformers import SentenceTransformer

# Recommended small and efficient model
MODEL_NAME = 'sentence-transformers/multi-qa-MiniLM-L6-cos-v1'
SAVE_DIR = './models/multi-qa-MiniLM-L6-cos-v1'  # Local folder to save model

def download_and_save():
    """
    Downloads the sentence-transformer model and saves it locally.
    Run this script once before building the Docker image to use the model offline.
    """
    print(f"Downloading model: {MODEL_NAME}")
    print("This may take a few minutes...")

    try:
        model = SentenceTransformer(MODEL_NAME)
        os.makedirs(os.path.dirname(SAVE_DIR), exist_ok=True)
        model.save(SAVE_DIR)

        print(f"\n Model downloaded and saved to: {SAVE_DIR}")
        print("You can now build and run the Docker image offline.")
    except Exception as e:
        print(f"\n Error downloading model: {e}")
        print("Please check your internet connection and try again.")

if __name__ == '__main__':
    download_and_save()