# Adobe Hackathon - Round 1B: Relevance Engine (Offline & Dockerized)

## Overview

This project builds on Round 1A’s **Structure Extractor** to implement a semantic **Relevance Engine**.  
Given a collection of **PDF documents**, a **persona**, and a **job-to-be-done** string, it returns a ranked list of the most relevant sections and refined sub‑sections.

- Runs entirely **offline**, on **CPU**
- Packaged in **Docker**
- Uses locally stored **sentence embedding models**

---

## ⚙Approach Explanation

### 🔹 Step 1: Structure Extraction

- Uses **StructureExtractor** to parse each PDF.
- Extracts headings (Title, H1, H2, H3) with page numbers.
- For each heading, extracts the full text **from that heading to the next** using **PyMuPDF**.

---

### 🔹 Step 2: Semantic Embedding

- Uses `multi-qa-MiniLM-L6-cos-v1` from **sentence-transformers**.
- Reasons for choosing:
  - Small size (~80 MB)
  - Fast CPU inference
  - High accuracy for semantic similarity
- Model is downloaded and included in the `models/` folder → **No internet required**.

---

### 🔹 Step 3: Query Construction & Section Encoding

- Combine persona and job:
  
  ```
  Persona: <persona>. Task: <job-to-be-done>
  ```
- Encode this query.
- Encode each extracted section’s text.

---

### 🔹 Step 4: Similarity Scoring & Ranking

- Compute **cosine similarity** between:
  - Query embedding
  - Section embeddings
- Collect metadata: `document`, `section_title`, `page_number`, `full_content`, `similarity_score`
- Sort descending by relevance → assign `importance_rank`.

---

### 🔹 Step 5: Sub‑Section Refinement (Optional)

- For top N sections:
  - Split into paragraphs or chunks.
  - Re-encode and re-rank.
  - Return the most relevant sub-section → `refined_text`.

---

### 🔹 Step 6: Output Assembly

Final output JSON includes:

- `document`
- `page_number`
- `section_title`
- `full_content`
- `relevance_score`
- `importance_rank`
- `refined_text` *(if available)*

---

## 🧾 Output Format Example

```json
[
  {
    "document": "Lunch Ideas.pdf",
    "page_number": 3,
    "section_title": "Quick Vegetarian Lunches",
    "full_content": "Text of the section…",
    "relevance_score": 0.92,
    "importance_rank": 1,
    "refined_text": "Try quinoa salad with roasted vegetables…"
  }
  // ...
]
```

---

## Requirements

**`requirements.txt`** contains:

```
sentence-transformers
torch
scipy
PyMuPDF
```

Install locally using:

```bash
pip install --no-cache-dir -r requirements.txt
```

---

## 🐳 Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "src/main_1b.py"]
```

---

## 🚀 Build & Run Instructions

1. Place your PDF files in `input/`.
2. Ensure the embedding model exists at `models/multi-qa-MiniLM-L6-cos-v1/`.
3. Build the Docker image:

```bash
docker build --platform linux/amd64 -t mysolution:1b .
```

4. Run the container:

```bash
docker run --rm -v $PWD/input:/app/input/ -v $PWD/output:/app/output --network none mysolution:1b
```

---
##  Project Structure

```
1b/
├── input/                         # Input PDF files
├── models/                        # Downloaded SentenceTransformer model
│   └── multi-qa-MiniLM-L6-cos-v1/ # Local embedding model files
├── output/                        # Output JSON files
├── src/
│   ├── main_1b.py                 # Pipeline entry point
│   ├── relevance_engine/
│   │   └── engine.py              # RelevanceEngine class
│   └── structure_extractor/
│       └── extractor.py           # StructureExtractor from Round 1A
├── requirements.txt               # Python dependencies
├── Dockerfile                     # CPU-only offline Docker execution
└── README.md                      # This file
```

---
## Constraints Met

- **Offline execution**: All models are pre-downloaded.
- **CPU-only**: Uses CPU version of PyTorch.
- **Fast inference**: Lightweight embedding model.
- **Dockerized**: Fully contained execution with no internet access.
