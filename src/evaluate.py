import time
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from src.preprocess import get_true_duplicate_pairs
from src.simhash import CustomSimHash

def compute_metrics(candidates: set, predicted: set, true_pairs: set, n_items: int):
    total = n_items * (n_items - 1) / 2
    frac_comp = len(candidates) / total if total else 0.0
    dups = len(candidates & true_pairs)
    pq = dups / len(candidates) if candidates else 0.0
    pc = dups / len(true_pairs) if true_pairs else 0.0
    f1_star = (2 * pq * pc / (pq + pc)) if (pq + pc) > 0 else 0.0

    tp = len(predicted & true_pairs)
    fp = len(predicted - true_pairs)
    fn = len(true_pairs - predicted)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    return {'frac_comp': frac_comp, 'PQ': pq, 'PC': pc, 'F1_star': f1_star, 'F1': f1}

def precompute_token_sets(products: list):
    alnum, inches, hzs = [], [], []
    for p in products:
        words = set(p['text'].split())
        inches.append({w for w in words if w.endswith('inch')})
        hzs.append({w for w in words if w.endswith('hz')})
        alnum.append({w for w in words if any(c.isalpha() for c in w) and any(c.isdigit() for c in w)
                      and not w.endswith(('inch', 'hz')) and w not in ('1080p', '720p', '480p', '1080i', '3d', '4k')})
    return alnum, inches, hzs

def pair_similarities(X, products: list, cand_list: list) -> np.ndarray:
    if not cand_list:
        return np.array([])
    rows = [i for i, _ in cand_list]
    cols = [j for _, j in cand_list]
    cos_sims = np.asarray(X[rows].multiply(X[cols]).sum(axis=1)).ravel()

    alnum, inches, hzs = precompute_token_sets(products)
    scores = np.zeros(len(cand_list), dtype=np.float64)
    for idx, (i, j) in enumerate(cand_list):
        if (inches[i] and inches[j] and not (inches[i] & inches[j])) or \
           (hzs[i] and hzs[j] and not (hzs[i] & hzs[j])):
            continue
        a1, a2 = alnum[i], alnum[j]
        mod_jac = len(a1 & a2) / len(a1 | a2) if (a1 | a2) else 0.0
        scores[idx] = 0.65 * mod_jac + 0.35 * cos_sims[idx]
    return scores

def cluster_pairs(n_items: int, cand_list: list, sims: np.ndarray, threshold: float, shops: list) -> set:
    if not cand_list:
        return set()
    active = [cand_list[k] for k in np.where(sims >= threshold)[0]]
    if not active:
        return set()
    r = [i for i, _ in active] + [j for _, j in active]
    c = [j for _, j in active] + [i for i, _ in active]
    adj = csr_matrix((np.ones(len(r), dtype=np.int8), (r, c)), shape=(n_items, n_items))
    _, labels = connected_components(adj, directed=False)
    return {(i, j) for i, j in cand_list if labels[i] == labels[j] and shops[i] != shops[j]}

def run_bootstrap_experiment(products: list, n_bootstraps: int = 5, num_bits: int = 240,
                             band_configs: list = None, ngram_range: tuple = (3, 4), seed: int = 42):
    if band_configs is None:
        band_configs = [30, 24, 20, 16]  # r = 8, 10, 12, 15 (keeps FracComp <= 0.11)

    rng = np.random.default_rng(seed)
    n_total = len(products)
    results = {b: [] for b in band_configs}
    t_start = time.perf_counter()

    for b_idx in range(n_bootstraps):
        train_idx = np.unique(rng.choice(n_total, size=n_total, replace=True))
        test_idx = np.setdiff1d(np.arange(n_total), train_idx)
        train_prods = [products[i] for i in train_idx]
        test_prods = [products[i] for i in test_idx]

        vec = TfidfVectorizer(analyzer='char_wb', ngram_range=ngram_range, sublinear_tf=True, min_df=2)
        X_train = vec.fit_transform([p['text'] for p in train_prods])
        X_test = vec.transform([p['text'] for p in test_prods])

        true_train = get_true_duplicate_pairs(train_prods)
        true_test = get_true_duplicate_pairs(test_prods)
        tr_shops = [p['shop'] for p in train_prods]
        te_shops = [p['shop'] for p in test_prods]

        simhash = CustomSimHash(num_bits=num_bits, seed=seed + b_idx)
        sig_train = simhash.fit_transform(X_train)
        sig_test = simhash.fit_transform(X_test)

        for bands in band_configs:
            tr_cands = list(simhash.get_candidate_pairs(sig_train, bands, tr_shops))
            tr_sims = pair_similarities(X_train, train_prods, tr_cands)

            best_th, best_f1 = 0.5, -1.0
            for th in np.linspace(0.35, 0.85, 21):
                pred_tr = cluster_pairs(len(train_prods), tr_cands, tr_sims, th, tr_shops)
                f1_tr = compute_metrics(set(tr_cands), pred_tr, true_train, len(train_prods))['F1']
                if f1_tr > best_f1:
                    best_f1, best_th = f1_tr, th

            te_cands = list(simhash.get_candidate_pairs(sig_test, bands, te_shops))
            te_sims = pair_similarities(X_test, test_prods, te_cands)
            pred_te = cluster_pairs(len(test_prods), te_cands, te_sims, best_th, te_shops)
            results[bands].append(compute_metrics(set(te_cands), pred_te, true_test, len(test_prods)))

        print(f"Bootstrap {b_idx + 1}/{n_bootstraps} finished ({time.perf_counter() - t_start:.1f}s)")

    return results