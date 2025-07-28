# src/relevance_engine/engine.py

import fitz
from pathlib import Path
from sentence_transformers import SentenceTransformer, util
from src.structure_extractor.extractor import StructureExtractor


class RelevanceEngine:
    def __init__(self, structure_extractor: StructureExtractor, input_dir, model_name: str = 'models/multi-qa-MiniLM-L6-cos-v1'):
        self.structure_extractor = structure_extractor
        self.device = 'cpu'
        print(f"RelevanceEngine using device: {self.device}")

        base_dir = Path(__file__).resolve().parent.parent.parent
        model_path = base_dir / model_name

        if not model_path.exists():
            raise RuntimeError(f"\n\nFATAL ERROR: A required model file was not found. Path {model_path} not found\nPlease ensure your model files exist at the specified paths.")

        self.model = SentenceTransformer(str(model_path), device=self.device)
        self.input_dir = input_dir

    def _get_section_content(self, doc: fitz.Document, outline: list, section_index: int) -> str:
        current_section = outline[section_index]
        start_page = current_section['page'] - 1
        start_y0 = 0
        page = doc[start_page]
        search_results = page.search_for(current_section['text'])
        if search_results:
            start_y0 = search_results[0].y0

        end_page = doc.page_count - 1
        end_y0 = page.rect.height
        if section_index + 1 < len(outline):
            next_section = outline[section_index + 1]
            if next_section['page'] > current_section['page']:
                end_page = next_section['page'] - 2
            else:
                end_page = start_page
                next_page = doc[next_section['page'] - 1]
                next_results = next_page.search_for(next_section['text'])
                if next_results:
                    end_y0 = next_results[0].y0

        content = ""
        for i in range(start_page, end_page + 1):
            page_to_extract = doc[i]
            y_start = start_y0 if i == start_page else 0
            y_end = end_y0 if i == end_page else page_to_extract.rect.height
            clip_rect = fitz.Rect(0, y_start, page_to_extract.rect.width, y_end)
            content += page_to_extract.get_text(clip=clip_rect)

        return content.strip()

    def rank_sections(self, pdf_paths: list, persona: str, job_to_be_done: str):
        query = f"Persona: {persona}. Task: {job_to_be_done}"
        print(f"Generating embeddings for query: '{query}'")
        query_embedding = self.model.encode(query, convert_to_tensor=True)

        all_sections_data = []
        doc_cache = {}  # To store open doc, structure, section_texts

        for pdf_path in pdf_paths:
            doc_name = Path(pdf_path).name
            print(f"\nProcessing document: {doc_name}")

            try:
                structure = self.structure_extractor.predict(pdf_path)
                outline = [entry for entry in structure.get('outline', []) if entry.get("level") == "H1"]
                if not outline:
                    print(f"  - No H1 outline found for {doc_name}, skipping.")
                    continue

                doc = fitz.open(pdf_path)
                section_contents = [self._get_section_content(doc, outline, i) for i in range(len(outline))]

                valid_indices = [i for i, content in enumerate(section_contents) if content]
                if not valid_indices:
                    continue

                valid_contents = [section_contents[i] for i in valid_indices]
                valid_outline = [outline[i] for i in valid_indices]

                print(f"  - Encoding {len(valid_contents)} sections...")
                section_embeddings = self.model.encode(valid_contents, convert_to_tensor=True, batch_size=16, show_progress_bar=False)

                similarities = util.cos_sim(query_embedding, section_embeddings)[0]

                for i, section_info in enumerate(valid_outline):
                    all_sections_data.append({
                        "document": doc_name,
                        "page_number": section_info['page'],
                        "section_title": section_info['text'],
                        "full_content": valid_contents[i],
                        "relevance_score": similarities[i].item()
                    })

                # Store cache for later
                doc_cache[doc_name] = {
                    "doc": doc,
                    "page_texts": [page.get_text() for page in doc]
                }

            except Exception as e:
                print(f"  ❌ Failed to process {doc_name}: {e}")
                continue

        ranked = sorted(all_sections_data, key=lambda x: x['relevance_score'], reverse=True)
        for i, section in enumerate(ranked):
            section['importance_rank'] = i + 1

        return ranked, doc_cache

    def analyze_subsections(self, ranked_sections, doc_cache, max_subsections=5):
        results = []
        for section in ranked_sections[:max_subsections]:
            doc_name = section["document"]
            page_num = section["page_number"] - 1

            try:
                if doc_name not in doc_cache:
                    print(f"⚠️ Document {doc_name} not found in cache.")
                    continue
                page_texts = doc_cache[doc_name]["page_texts"]
                if page_num < 0 or page_num >= len(page_texts):
                    continue

                lines = [line.strip() for line in page_texts[page_num].split('\n') if line.strip()]
                refined_text = ' '.join(lines[:2]) if lines else ""

                results.append({
                    "document": doc_name,
                    "page_number": page_num + 1,
                    "refined_text": refined_text
                })

            except Exception as e:
                print(f"❌ Error during subsection analysis for {doc_name}: {e}")

        return results
