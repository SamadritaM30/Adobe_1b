# src/main_1b.py

import json
import sys
from pathlib import Path
from datetime import datetime, timezone

# --- Add 'src' to the Python path for imports ---
try:
    SRC_DIR = Path(__file__).resolve().parent
    sys.path.append(str(SRC_DIR.parent))
except NameError:
    sys.path.append(str(Path('.').resolve()))

from structure_extractor.extractor import StructureExtractor
from relevance_engine.engine import RelevanceEngine

# --- Configuration ---
PERSONA = "Food Contractor"
JOB = "Prepare a vegetarian buffet-style dinner menu for a corporate gathering, including gluten-free items."

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_DIR = PROJECT_ROOT / 'input'
OUTPUT_DIR = PROJECT_ROOT / 'output'
MODELS_DIR = PROJECT_ROOT / 'models'

STRUCTURE_MODEL_PATH = MODELS_DIR / 'heading_classifier.pkl'
STRUCTURE_ARTIFACTS_PATH = MODELS_DIR / 'model_artifacts.json' 

def run_1b_pipeline():
    print("--- Starting Adobe Hackathon - Round 1B Pipeline ---")

    try:
        structure_extractor = StructureExtractor(
            model_path=str(STRUCTURE_MODEL_PATH),
            artifacts_path=str(STRUCTURE_ARTIFACTS_PATH)
        )
        relevance_engine = RelevanceEngine(structure_extractor, input_dir=INPUT_DIR)
    except FileNotFoundError as e:
        print(f"\nFATAL ERROR: A required model file was not found. {e}")
        return

    input_json_path = INPUT_DIR / 'challenge1b_input.json'
    if not input_json_path.exists():
        print(f"Error: Input JSON file not found at: {input_json_path}")
        return

    with open(input_json_path, 'r', encoding='utf-8') as f:
        input_data = json.load(f)

    requested_filenames = [doc["filename"] for doc in input_data.get("documents", [])]
    pdf_paths = [str(INPUT_DIR / f) for f in requested_filenames if (INPUT_DIR / f).exists()]
    missing_files = [f for f in requested_filenames if not (INPUT_DIR / f).exists()]

    if missing_files:
        print(f"Warning: Missing files listed in JSON but not found: {missing_files}")
    if not pdf_paths:
        print("Error: No valid PDF files found. Exiting.")
        return

    print(f"Found {len(pdf_paths)} documents listed in JSON for analysis: {requested_filenames}")

    # --- Core Logic ---
    query = f"Persona: {PERSONA}. Task: {JOB}"
    ranked_sections, doc_cache = relevance_engine.rank_sections(pdf_paths, PERSONA, JOB)
    top_sections = ranked_sections[:5]

    sub_section_analysis = relevance_engine.analyze_subsections(top_sections, doc_cache, max_subsections=5)

    output_data = {
        "metadata": {
            "input_documents": requested_filenames,
            "persona": PERSONA,
            "job_to_be_done": JOB,
            "processing_timestamp": datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
        },
        "extracted_section": [
            {
                "document": s['document'],
                "page_number": s['page_number'],
                "section_title": s['section_title'],
                "importance_rank": s['importance_rank']
            } for s in top_sections
        ],
        "subsection_analysis": sub_section_analysis
    }

    OUTPUT_DIR.mkdir(exist_ok=True)
    output_path = OUTPUT_DIR / 'challenge1b_output.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=4)

    print(f"\n✅ Pipeline finished. Output saved to: {output_path}")

if __name__ == '__main__':
    run_1b_pipeline()