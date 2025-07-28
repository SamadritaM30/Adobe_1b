import os
from structure_extractor.extractor import StructureExtractor
import json

# Get the directory of the current script to build relative paths
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)  # Assumes script is in 'src'

INPUT_DIR = os.path.join(project_root, "input")
OUTPUT_DIR = os.path.join(project_root, "output")
MODEL_PATH = os.path.join(project_root, "models", "heading_classifier.pkl")
ARTIFACTS_PATH = os.path.join(project_root, "models", "model_artifacts.json")

if __name__ == "__main__":
    extractor = StructureExtractor(MODEL_PATH, ARTIFACTS_PATH)
    for fname in os.listdir(INPUT_DIR):
        if fname.lower().endswith(".pdf"):
            pdf_path = os.path.join(INPUT_DIR, fname)
            result_json = extractor.predict(pdf_path) # Changed from extract to predict
            out_path = os.path.join(OUTPUT_DIR, fname.replace(".pdf", ".json"))
            with open(out_path, "w") as f:
                json.dump(result_json, f, indent=2) 