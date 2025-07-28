# src/structure_extractor/features.py

import pandas as pd
import re

def engineer_features(df: pd.DataFrame, body_text_size: float, page_width: float = 612.0) -> pd.DataFrame:
    if df.empty:
        return df

    # Ensure bbox is a tuple
    if isinstance(df['bbox'].iloc[0], str):
        df['bbox'] = df['bbox'].apply(lambda x: tuple(map(float, re.findall(r'[\d\.]+', x))))

    df['x0'] = df['bbox'].apply(lambda bbox: bbox[0])
    df['y0'] = df['bbox'].apply(lambda bbox: bbox[1])
    df['x1'] = df['bbox'].apply(lambda bbox: bbox[2])
    df['y1'] = df['bbox'].apply(lambda bbox: bbox[3])

    df['relative_font_size'] = df['font_size'] / body_text_size
    df['is_bold'] = df['font_name'].str.contains('bold', case=False, na=False).astype(int)
    df['is_italic'] = df['font_name'].str.contains('italic', case=False, na=False).astype(int)
    df['text_length'] = df['text'].fillna("").str.len()
    df['word_count'] = df['text'].fillna("").str.split().str.len()
    df['is_all_caps'] = df['text'].fillna("").apply(lambda x: x.isupper() and len(x) > 1).astype(int)
    df['contains_colon'] = df['text'].fillna("").str.contains(":").astype(int)
    df['ends_with_punctuation'] = df['text'].fillna("").str.contains(r'[.!?]$').astype(int)

    numbering_pattern = r'^\s*((Appendix\s[A-Z])|(\d+(\.\d+)*)|([A-Z]\.))\s*'
    df['starts_with_numbering'] = df['text'].fillna("").str.match(numbering_pattern, flags=re.IGNORECASE).astype(int)

    block_width = df['x1'] - df['x0']
    block_center = df['x0'] + (block_width / 2)
    df['is_centered'] = (abs((page_width / 2) - block_center) < 20).astype(int)

    if 'page_num' not in df.columns and 'page' in df.columns:
        df['page_num'] = df['page']
    df_sorted = df.sort_values(by=['page_num', 'y0'])
    prev_y1 = df_sorted.groupby('page_num')['y1'].shift(1)
    df['space_before'] = df['y0'] - prev_y1
    df['space_before'] = df['space_before'].fillna(df['y0'])

    df['normalized_y0'] = df['y0'] / df['page_height']
    return df

FEATURE_COLUMNS = [
    'relative_font_size',
    'is_bold',
    'is_italic',
    'text_length',
    'word_count',
    'is_all_caps',
    'contains_colon',
    'ends_with_punctuation',
    'starts_with_numbering',
    'is_centered',
    'space_before',
    'normalized_y0'
]
