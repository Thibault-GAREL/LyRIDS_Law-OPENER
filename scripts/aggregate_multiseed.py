"""Agrège les résultats multi-seed OPENER-Legal en mean ± std par cellule.

Contexte : le paper reporte OPENER avec l'embedder « released » (seed 42 du
papier principal). L'étude multi-seed réévalue les mêmes têtes avec les
embedders ré-entraînés des seeds 7 et 123 du repo principal
(../LyRIDS_Opener/outputs/models/seed{S}_hard), sans rien ré-entraîner ici.
Ce script rassemble les trois seeds et produit, pour chaque
(tête, dataset, métrique) : per-seed, mean, std (ddof=1).

Têtes agrégées (protocoles EXACTS du paper, vérifiés contre ses tables) :
    sup_gold  results[ds][metric]['linear_svm_balanced']       (balanced_classifiers)
    zs_gold   results[ds][metric]                              (opener_zs dict, nearest centroid)
    sup_e2e   results[ds]['by_threshold']['0.3'][metric][SVM]  (opener_e2e GLiNER-M fit-on-detected)
    zs_e2e    results[ds]['transductive']['0.05'][metric]      (zs_fusion GLiNER-L, beta 0.05)

Seed 42 = les runs canoniques du paper (SEED42_SOURCES, fusion multi-fichiers
car les datasets ont été lancés par vagues). Seeds 7/123 =
outputs/results/seed_runs/seed{S}/{sup_gold,zs_gold,sup_e2e,zs_e2e}/{dataset}/.

Convention Avg : moyenne benchmark par seed d'abord, puis mean±std de ces moyennes.

Usage :
    python -m scripts.aggregate_multiseed --check    # valide seed 42 contre le paper (à faire AVANT les runs)
    python -m scripts.aggregate_multiseed            # agrège les seeds disponibles
    python -m scripts.aggregate_multiseed --latex    # + lignes LaTeX (cellule "35.5$_{\\pm0.3}$")
"""
import argparse
import glob
import json
import statistics
from pathlib import Path

DATASETS = ['e_ner', 'indian_legal', 'lener_br', 'german_ler_coarse']
STRESS_DATASETS = ['german_ler']          # schéma fin 19 types, têtes gold uniquement
METRICS = ['ami', 'macro_f1', 'accuracy']
SVM = 'linear_svm_balanced'
ALL_SEEDS = [42, 7, 123]                  # 42 = released (runs canoniques du paper)

O = 'outputs/results'

# Runs canoniques du paper (seed 42). Chaque tête fusionne tous ses SUMMARY
# (un run par vague de datasets) ; le fichier le plus récent gagne par dataset.
SEED42_SOURCES = {
    'sup_gold': f'{O}/legal/sup_gold/SUMMARY_balanced_classifiers_*.json',
    'zs_gold':  f'{O}/legal/zs_gold/SUMMARY_opener_zs_dict_*.json',
    'sup_e2e':  f'{O}/legal/sup_e2e_glinerM_det/SUMMARY_opener_e2e_*.json',
    'zs_e2e':   f'{O}/legal/zs_e2e_fusion/SUMMARY_zs_fusion_*.json',
}

# tête -> extracteur(results[ds], metric)
HEADS = {
    'sup_gold': lambda r, m: r[m][SVM],
    'zs_gold':  lambda r, m: r[m],
    'sup_e2e':  lambda r, m: r['by_threshold']['0.3'][m][SVM],
    'zs_e2e':   lambda r, m: r['transductive']['0.05'][m],
}
GOLD_HEADS = ['sup_gold', 'zs_gold']

# Cellules AMI (x100) des tables du paper pour la validation --check.
PAPER_CELLS = {
    'sup_gold': {'e_ner': 70.0, 'indian_legal': 61.9, 'lener_br': 53.3,
                 'german_ler_coarse': 59.1, 'german_ler': 62.0},
    'zs_gold':  {'e_ner': 44.0, 'indian_legal': 46.3, 'lener_br': 31.6,
                 'german_ler_coarse': 22.6, 'german_ler': 22.1},
    'sup_e2e':  {'e_ner': 35.5, 'indian_legal': 35.4, 'lener_br': 34.2,
                 'german_ler_coarse': 38.4},
    'zs_e2e':   {'e_ner': 33.2, 'indian_legal': 34.0, 'lener_br': 34.6,
                 'german_ler_coarse': 33.2},
}
# Convention du paper : les moyennes affichées suivent l'arithmétique des
# cellules arrondies (un reviewer qui recalcule depuis la table retombe dessus).
PAPER_MEANS = {'sup_gold': 61.1, 'zs_gold': 36.1, 'sup_e2e': 35.9, 'zs_e2e': 33.8}


def _merge_results(pattern):
    """Fusionne les `results` de tous les SUMMARY matchant `pattern`.
    En cas de doublon pour un dataset, le fichier le plus récent gagne
    (ordre lexicographique des noms horodatés)."""
    merged = {}
    for f in sorted(glob.glob(pattern)):
        merged.update(json.load(open(f, encoding='utf-8')).get('results', {}))
    return merged


def load_seed(seed):
    """Retourne {head: results_merged} pour un seed (None si tête absente)."""
    out = {}
    for head in HEADS:
        if seed == 42:
            pattern = SEED42_SOURCES[head]
        else:
            pattern = f'{O}/seed_runs/seed{seed}/{head}/*/SUMMARY*.json'
        res = _merge_results(pattern)
        out[head] = res or None
    return out


def extract(per_seed, head, ds, metric):
    """Valeur (x100) pour (seed déjà chargé, tête, dataset, métrique), ou None."""
    res = per_seed.get(head)
    if not res or ds not in res:
        return None
    try:
        return round(100 * HEADS[head](res[ds], metric), 10)
    except (KeyError, TypeError):
        return None


def check_seed42():
    """Étape 0.3 : l'extraction seed 42 doit reproduire les tables du paper."""
    per_seed = load_seed(42)
    n_fail = 0
    for head, cells in PAPER_CELLS.items():
        vals = []
        for ds, expected in cells.items():
            got = extract(per_seed, head, ds, 'ami')
            got_r = None if got is None else round(got, 1)
            ok = got_r == expected
            n_fail += not ok
            print(f"  {'OK ' if ok else 'FAIL'} {head:8s} {ds:18s} paper={expected:5.1f} extrait={got_r}")
            if ds in DATASETS:
                vals.append(got_r)
        mean = round(sum(vals) / len(vals), 1)
        ok = mean == PAPER_MEANS[head]
        n_fail += not ok
        print(f"  {'OK ' if ok else 'FAIL'} {head:8s} {'MEAN(4 ds, cellules arrondies)':18s} paper={PAPER_MEANS[head]:5.1f} extrait={mean}")
    print('\nVALIDATION SEED 42 :', 'PASS (tables du paper reproduites cellule par cellule)'
          if n_fail == 0 else f'{n_fail} CELLULE(S) EN ECHEC, ne PAS lancer les runs')
    return n_fail == 0


def aggregate(latex=False):
    data = {s: load_seed(s) for s in ALL_SEEDS}
    seeds_present = [s for s in ALL_SEEDS
                     if any(data[s][h] for h in HEADS)]
    agg = {}
    for head in HEADS:
        dss = DATASETS + (STRESS_DATASETS if head in GOLD_HEADS else [])
        agg[head] = {}
        for m in METRICS:
            agg[head][m] = {}
            for ds in dss:
                per = {s: extract(data[s], head, ds, m) for s in seeds_present}
                vals = [v for v in per.values() if v is not None]
                if not vals:
                    continue
                agg[head][m][ds] = {
                    'per_seed': per,
                    'mean': statistics.mean(vals),
                    'std': statistics.stdev(vals) if len(vals) > 1 else 0.0,
                    'n_seeds': len(vals),
                }
            # Colonne Avg : moyenne benchmark par seed, puis mean/std de ces moyennes.
            per_seed_avgs = {}
            for s in seeds_present:
                svals = [extract(data[s], head, ds, m) for ds in DATASETS]
                if all(v is not None for v in svals):
                    per_seed_avgs[s] = statistics.mean(svals)
            if per_seed_avgs:
                v = list(per_seed_avgs.values())
                agg[head][m]['__avg__'] = {
                    'per_seed': per_seed_avgs,
                    'mean': statistics.mean(v),
                    'std': statistics.stdev(v) if len(v) > 1 else 0.0,
                    'n_seeds': len(v),
                }

    outdir = Path(f'{O}/aggregate')
    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / 'results_multiseed.json'
    json.dump({'seeds': seeds_present, 'datasets': DATASETS,
               'stress_datasets': STRESS_DATASETS, 'agg': agg},
              open(out, 'w', encoding='utf-8'), indent=2)
    print(f'Seeds agrégés : {seeds_present} -> {out}')

    def cell(head, ds, m='ami'):
        e = agg[head][m].get(ds)
        if e is None:
            return 'n/a'
        return f"{e['mean']:.1f}$_{{\\pm{e['std']:.1f}}}$" if e['n_seeds'] > 1 else f"{e['mean']:.1f}"

    print('\n## AMI (x100), mean ± std sur les seeds', seeds_present)
    hdr = ['head'] + DATASETS + ['Avg']
    print('| ' + ' | '.join(hdr) + ' |')
    print('|' + '---|' * len(hdr))
    for head in HEADS:
        row = [head] + [cell(head, ds).replace('$_{\\pm', ' ±').replace('}$', '')
                        for ds in DATASETS + ['__avg__']]
        print('| ' + ' | '.join(row) + ' |')
    for head in GOLD_HEADS:
        print(f'| {head} (german_ler fine) | ' +
              cell(head, 'german_ler').replace('$_{\\pm', ' ±').replace('}$', '') + ' |')

    if latex:
        print('\n## Lignes LaTeX (tab:legal-gold puis tab:legal-main, cellules AMI x100)')
        for head, label in [('zs_gold', 'OPENER-ZS'), ('sup_gold', 'OPENER-Sup')]:
            cells = [cell(head, ds) for ds in DATASETS + ['__avg__']]
            print(f'\\rowcolor{{...}} {label} & ' + ' & '.join(cells) + ' \\\\')
        for head, label in [('zs_e2e', 'OPENER-ZS'), ('sup_e2e', 'OPENER-Sup')]:
            cells = [cell(head, ds) for ds in DATASETS + ['__avg__']]
            print(f'\\rowcolor{{...}} {label} & ' + ' & '.join(cells) + ' \\\\')
    return agg


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--check', action='store_true',
                    help='Valide la reproduction des tables du paper avec le seul seed 42')
    ap.add_argument('--latex', action='store_true')
    args = ap.parse_args()
    if args.check:
        raise SystemExit(0 if check_seed42() else 1)
    aggregate(latex=args.latex)


if __name__ == '__main__':
    main()
