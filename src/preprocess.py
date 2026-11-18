import json
import re
from itertools import combinations

IMPORTANT_KEYS = ('brand', 'screen size', 'resolution', 'refresh rate', 'display', 'tv type') # add aliases?

def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'(\d+)\s*(hz|hertz)', r'\1hz', text) # standardize refresh rate
    text = re.sub(r'(\d+(?:\.\d+)?)\s*(?:"|inch|inches|-inch)', r'\1inch', text) # standardize screen size
    text = re.sub(r'(\d+)\s*x\s*(\d+)', r'\1x\2', text) # remove resolution spaces
    text = re.sub(r'([a-z0-9])[-/]([a-z0-9])', r'\1\2', text) # remove hyphens and slashes between alphanumeric characters
    text = re.sub(r'[^a-z0-9\s]', ' ', text) # replace non-alphanumeric characters with spaces
    return re.sub(r'\s+', ' ', text).strip() # collapse multiple spaces and trim

def load_dataset(filepath: str, title_weight: int = 3):
    with open(filepath, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    products = []
    for model_id, items in raw_data.items():
        for item in items:
            title = normalize_text(item.get('title', ''))
            kv = [normalize_text(str(v)) for k, v in item.get('featuresMap', {}).items()
                  if any(key in k.lower() for key in IMPORTANT_KEYS)]
            products.append({
                'modelID': model_id,
                'shop': item.get('shop', ''),
                'text': ' '.join([title] * title_weight + kv).strip()
            })
    return products

def get_true_duplicate_pairs(products: list) -> set:
    return {(i, j) for (i, p1), (j, p2) in combinations(enumerate(products), 2)
            if p1['modelID'] == p2['modelID']}