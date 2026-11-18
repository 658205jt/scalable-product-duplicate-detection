import time
import numpy as np
from src.preprocess import load_dataset
from src.evaluate import run_bootstrap_experiment
from src.plot_results import save_results

if __name__ == '__main__':
    t0 = time.perf_counter()
    products = load_dataset('data/TVs-all-merged.json', title_weight=3)
    print(f"Loaded {len(products)} products in {time.perf_counter() - t0:.2f}s.")

    num_bits = 240
    band_configs = [48, 40, 30, 24, 20, 16]
    results = run_bootstrap_experiment(
        products, n_bootstraps=5, num_bits=num_bits, band_configs=band_configs
    )

    print(f"\n{'Bands':>5} | {'Rows':>4} | {'FracComp':>8} | {'PQ':>6} | {'PC':>6} | {'F1*':>6} | {'F1':>6}")
    print("-" * 55)
    for b in band_configs:
        avg = {k: np.mean([m[k] for m in results[b]]) for k in results[b][0]}
        print(f"{b:5d} | {num_bits//b:4d} | {avg['frac_comp']:8.4f} | {avg['PQ']:6.4f} | "
              f"{avg['PC']:6.4f} | {avg['F1_star']:6.4f} | {avg['F1']:6.4f}")

    save_results(results, band_configs, num_bits, out_dir='results')