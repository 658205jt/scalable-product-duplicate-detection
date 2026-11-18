import numpy as np
from collections import defaultdict
from itertools import combinations

class CustomSimHash:
    def __init__(self, num_bits: int = 128, seed: int = 42):
        self.num_bits = num_bits
        self.rng = np.random.default_rng(seed)
        self.hyperplanes = None

    def fit_transform(self, tfidf_matrix) -> np.ndarray:
        n_features = tfidf_matrix.shape[1]
        if self.hyperplanes is None or self.hyperplanes.shape[1] != n_features:
            self.hyperplanes = self.rng.standard_normal((self.num_bits, n_features))
        # Sparse dot product: (N, D) @ (D, K) -> (N, K)
        projections = tfidf_matrix.dot(self.hyperplanes.T)
        return (projections >= 0).astype(np.uint8)

    def get_candidate_pairs(self, signatures: np.ndarray, bands: int, shops: list = None) -> set:
        n_items, n_bits = signatures.shape
        assert n_bits % bands == 0, "num_bits must be divisible by bands"
        rows = n_bits // bands
        candidates = set()

        for b in range(bands):
            buckets = defaultdict(list)
            start, end = b * rows, (b + 1) * rows
            for idx in range(n_items):
                band_hash = signatures[idx, start:end].tobytes()
                buckets[band_hash].append(idx)

            for bucket_items in buckets.values():
                if len(bucket_items) > 1:
                    for i, j in combinations(sorted(bucket_items), 2):
                        if shops is None or shops[i] != shops[j]:
                            candidates.add((i, j))
        return candidates