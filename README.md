```markdown
# Adobe Hackathon - Round 1B: Relevance Engine (Offline & Dockerized)

## Overview
This project builds on Round 1A’s Structure Extractor to implement a semantic Relevance Engine. Given a collection of PDF documents, a persona, and a job‑to‑be‑done string, it returns a ranked list of the most relevant sections and refined sub‑sections. The pipeline runs entirely offline, on CPU, and is packaged in Docker.

## Project Structure
```

1b/
├── input/                             # Directory containing input PDF files
├── models/                            # Folder containing downloaded SentenceTransformer model
│   └── multi-qa-MiniLM-L6-cos-v1/     # Local embedding model files
├── output/                            # Generated output JSON files
├── src/
│   ├── main\_1b.py                     # Entry point for the pipeline
│   ├── relevance\_engine/
│   │   └── engine.py                  # RelevanceEngine class with ranking logic
│   └── structure\_extractor/
│       └── extractor.py               # StructureExtractor from Round 1A
├── requirements.txt                   # Python dependencies
├── Dockerfile                         # Dockerfile for CPU‑only offline execution
└── README.md                          # This file

```

## Approach Explanation
### Step 1: Structure Extraction  
Reuse the StructureExtractor from Round 1A to parse each PDF and extract its outline (Title, H1, H2, H3) with page numbers. For each heading, we extract the full text from that heading to the next heading using PyMuPDF.

### Step 2: Semantic Embedding  
To capture relevance, we convert text into dense vectors using the `multi-qa-MiniLM-L6-cos-v1` model from the sentence-transformers library. This model is chosen for its small size (~80 MB), fast CPU inference, and high accuracy in semantic similarity tasks. We download it locally and include it in the `models/` folder so that no internet access is required at runtime.

### Step 3: Query Construction and Section Encoding  
We concatenate persona and job into a single query string:  
```

"Persona: <persona>. Task: <job-to-be-done>"

````
We encode this query into an embedding. For each extracted section, we encode its text into an embedding.

### Step 4: Similarity Scoring and Ranking  
Compute cosine similarity between the query embedding and each section embedding using `scipy.spatial.distance.cosine`. Lower cosine distance indicates higher relevance. We collect metadata (document name, section title, page number, full text) and similarity scores, then sort all sections in descending order of relevance to produce an `importance_rank` for each.

### Step 5: Sub‑Section Refinement  
Optionally, we refine within the top N sections by chunking their text into paragraphs or fixed‐length blocks, re‑encoding and re‑ranking those chunks to extract a `refined_text`—the most relevant sub‑section content.

### Step 6: Output Assembly  
We assemble a JSON array of ranked sections and refined sub‑sections, each entry containing:
- `document`
- `page_number`
- `section_title`
- `full_content`
- `relevance_score`
- `importance_rank`
- `refined_text` (if available)

## Output Format Example  
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
  },
  ...
]
````

## Requirements

Contents of `requirements.txt`:

```
sentence-transformers
torch
scipy
PyMuPDF
```

Install locally with:

```
pip install --no-cache-dir -r requirements.txt
```

## Dockerfile

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir -r requirements.txt
CMD ["python", "src/main_1b.py"]
```

## Build and Run Instructions

1. Place your PDF files in `input/`.
2. Ensure the embedding model folder `models/multi-qa-MiniLM-L6-cos-v1` is present.
3. Build the Docker image:

   ```
   docker build --platform linux/amd64 -t mysolution:1b .
   ```
4. Run the container:

   ```
    docker run --rm -v $PWD/input:/app/input/ -v $PWD/output:/app/output --network none mysolution:1b
   ```

## Constraints Met

* Offline execution: All models pre‑downloaded and bundled locally.
* CPU‑only: Uses CPU version of PyTorch.
* Fast inference: Small embedding model.
* Dockerized: Runs fully in container with no network access.

```
```


