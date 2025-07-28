# src/structure_extractor/extractor.py

import pandas as pd
import fitz  # PyMuPDF
import joblib
import json
from .features import engineer_features

class StructureExtractor:
    """
    Extracts the hierarchical structure (Title, H1, H2, H3) from a PDF file
    using a pre-trained machine learning model.
    """
    def __init__(self, model_path: str, artifacts_path: str):
        print("Loading model and artifacts...")
        self.model = joblib.load(model_path)
        with open(artifacts_path, 'r') as f:
            artifacts = json.load(f)
        self.body_text_size = artifacts['body_text_size']
        self.feature_columns = artifacts['feature_columns']
        print("StructureExtractor initialized.")

    def _parse_pdf_to_blocks(self, pdf_path: str) -> pd.DataFrame:
        doc = fitz.open(pdf_path)
        blocks_data = []
        for page_num, page in enumerate(doc):
            page_blocks = page.get_text("dict")["blocks"]
            page_height = page.rect.height
            page_width = page.rect.width
            for block in page_blocks:
                if block['type'] == 0:
                    for line in block["lines"]:
                        for span in line["spans"]:
                            blocks_data.append({
                                'text': span['text'].strip(),
                                'font_size': span['size'],
                                'font_name': span['font'],
                                'bbox': span['bbox'],
                                'x0': span['bbox'][0],
                                'y0': span['bbox'][1],
                                'x1': span['bbox'][2],
                                'y1': span['bbox'][3],
                                'page_num': page_num + 1,
                                'page_height': page_height,
                                'page_width': page_width
                            })
        doc.close()
        return pd.DataFrame(blocks_data)

    def predict(self, pdf_path: str) -> dict:
        print(f"Processing {pdf_path}...")
        df = self._parse_pdf_to_blocks(pdf_path)
        if df.empty:
            print("No text blocks found in PDF.")
            return {"title": "", "outline": []}
        print(f"Found {len(df)} text blocks.")

        page_width = df['page_width'].iloc[0] if not df.empty else 612.0
        df = engineer_features(df, self.body_text_size, page_width=page_width)
        print("Feature engineering complete.")
        print("Columns after feature engineering:", df.columns.tolist())
        print("Feature columns expected by model:", self.feature_columns)

        missing_cols = set(self.feature_columns) - set(df.columns)
        if missing_cols:
            print(f"Missing columns in dataframe: {missing_cols}")
            for c in missing_cols:
                df[c] = 0

        X_predict = df[self.feature_columns]

        print("Predicting labels...")
        predictions = self.model.predict(X_predict)
        df['predicted_label'] = predictions
        print(f"Predictions made. Unique labels found: {df['predicted_label'].unique()}")

        print("\nPrediction summary:")
        print(df['predicted_label'].value_counts())

        print("\nSample predictions:")
        print(df[['text', 'predicted_label']].head(10))

        headings_df = df[df['predicted_label'].isin(['Title', 'H1', 'H2', 'H3', 'H4'])]
        print(f"Found {len(headings_df)} headings.")
        sorted_headings = headings_df.sort_values(by=['page_num', 'y0'])

        output = {"title": "", "outline": []}

        title_rows = sorted_headings[sorted_headings['predicted_label'] == 'Title']
        if not title_rows.empty:
            output['title'] = title_rows.iloc[0]['text']

        outline_rows = sorted_headings[sorted_headings['predicted_label'].isin(['H1', 'H2', 'H3'])]
        for _, row in outline_rows.iterrows():
            output['outline'].append({
                "level": row['predicted_label'],
                "text": row['text'],
                "page": int(row['page_num'])
            })

        return output
