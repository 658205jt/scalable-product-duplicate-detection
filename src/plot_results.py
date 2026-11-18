import os
import csv
import numpy as np
import matplotlib.pyplot as plt

def save_results(results: dict, band_configs: list, num_bits: int, out_dir: str = 'results'):
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for b in sorted(band_configs, reverse=True):
        m = results[b]
        rows.append({
            'b': b, 'r': num_bits // b,
            'frac_comp': np.mean([x['frac_comp'] for x in m]),
            'PQ': np.mean([x['PQ'] for x in m]),
            'PC': np.mean([x['PC'] for x in m]),
            'F1_star': np.mean([x['F1_star'] for x in m]),
            'F1': np.mean([x['F1'] for x in m]),
        })

    # Save CSV
    csv_path = os.path.join(out_dir, 'metrics.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    # Save LaTeX table
    tex_path = os.path.join(out_dir, 'table.tex')
    with open(tex_path, 'w') as f:
        f.write("\\begin{table}[t]\n\\centering\n")
        f.write("\\caption{Average out-of-sample LSH blocking and duplicate detection results across 5 bootstraps ($K=240$ bits).}\n")
        f.write("\\label{tab:results}\n")
        f.write("\\begin{tabular}{rrccccc}\n\\hline\n")
        f.write("$b$ & $r$ & Frac. Comp. & $PQ$ & $PC$ & $F_1^*$ & $F_1$ \\\\\n\\hline\n")
        for r in rows:
            f.write(f"{r['b']} & {r['r']} & {r['frac_comp']:.4f} & {r['PQ']:.4f} & "
                    f"{r['PC']:.4f} & {r['F1_star']:.4f} & {r['F1']:.4f} \\\\\n")
        f.write("\\hline\n\\end{tabular}\n\\end{table}\n")

    # Sort ascending by fraction of comparisons for clean line plots
    rows_asc = sorted(rows, key=lambda x: x['frac_comp'])
    fc = [r['frac_comp'] for r in rows_asc]

    plt.rcParams.update({'font.size': 10, 'axes.grid': True, 'grid.alpha': 0.3})
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), dpi=300)

    # Panel 1: Pair Completeness & Final F1 vs FracComp
    axes[0].plot(fc, [r['PC'] for r in rows_asc], 'o-', label='Pair Completeness ($PC$)', color='#1f77b4')
    axes[0].plot(fc, [r['F1'] for r in rows_asc], 's--', label='Duplicate Detection $F_1$', color='#2ca02c')
    axes[0].set_xlabel('Fraction of Comparisons')
    axes[0].set_ylabel('Score')
    axes[0].set_title('Completeness & $F_1$ vs. Comparisons')
    axes[0].set_ylim(0, 1.05)
    axes[0].legend(loc='lower right', fontsize=8)

    # Panel 2: Pair Quality & F1* vs FracComp (log x-axis highlights low-comp regime)
    axes[1].plot(fc, [r['PQ'] for r in rows_asc], 'o-', label='Pair Quality ($PQ$)', color='#d62728')
    axes[1].plot(fc, [r['F1_star'] for r in rows_asc], 's--', label='LSH Harmonic Mean ($F_1^*$)', color='#ff7f0e')
    axes[1].set_xscale('log')
    axes[1].set_xlabel('Fraction of Comparisons (log scale)')
    axes[1].set_ylabel('Score')
    axes[1].set_title('Blocking Efficiency ($PQ$ & $F_1^*$)')
    axes[1].legend(loc='upper right', fontsize=8)

    # Panel 3: Zoomed F1 efficiency frontier (0 to 15% comparisons)
    axes[2].plot(fc, [r['F1'] for r in rows_asc], 's-', color='#2ca02c', linewidth=2)
    for r in rows_asc:
        axes[2].annotate(f"b={r['b']}", (r['frac_comp'], r['F1']),
                         textcoords="offset points", xytext=(0, 6), ha='center', fontsize=8)
    axes[2].set_xlabel('Fraction of Comparisons')
    axes[2].set_ylabel('Final $F_1$ Score')
    axes[2].set_title('Effectiveness–Efficiency Frontier')
    axes[2].set_ylim(0.20, 0.60)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'lsh_curves.pdf'))
    plt.savefig(os.path.join(out_dir, 'lsh_curves.png'))
    plt.close(fig)
    print(f"\nSaved metrics.csv, table.tex, and lsh_curves.pdf/.png to '{out_dir}/'")