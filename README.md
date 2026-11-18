Markdown
# Scalable Product Duplicate Detection via Character N-Gram TF-IDF and SimHash

Implementation of a scalable two-stage product duplicate detection pipeline evaluated on the `TVs-all-merged` dataset.

## Method Overview
1. **Product Representation:** Text normalization (unit standardization for inches/Hz/dimensions and hyphen collapsing inside model codes) over weighted product titles and core attributes, vectorized using sublinear word-boundary character $n$-grams ($n \in [3, 4]$).
2. **Stage 1 — Custom SimHash LSH (`src/simhash.py`):** From-scratch 240-bit Random Hyperplane Locality-Sensitive Hashing partitioned into $b$ bands of $r = 240/b$ rows to filter candidate pairs across distinct web shops.
3. **Stage 2 — Hybrid Pairwise Clustering (`src/evaluate.py`):** Closed-form similarity combining alphanumeric model-token Jaccard and character $n$-gram cosine similarity (with brand, screen-size, and refresh-rate mismatch filtering) followed by connected-components clustering.

## Setup & Reproduction
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python main.py