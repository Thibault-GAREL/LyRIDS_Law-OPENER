"""Vérification mécanique des chiffres du paper contre l'agrégation multi-seed.

Trois passes (étape 4 du protocole multi-seed) :
  1. Parse chaque cellule OPENER "xx.x$_{\\pm y.y}$" des tables de
     04_experiments.tex et la compare au JSON d'agrégation
     (outputs/results/aggregate/results_multiseed.json), mean et std arrondis.
  2. Recalcule les chiffres dérivés cités dans le texte (moyennes, plages,
     bornes de stabilité) depuis les CELLULES AFFICHÉES (convention du paper).
  3. Liste toutes les phrases contenant « seed » pour relecture humaine.

Usage : python -m scripts.verify_paper_numbers
Sortie : OK/FAIL par vérification, exit 1 si au moins un FAIL.
"""
import json
import re
from pathlib import Path

AGG = 'outputs/results/aggregate/results_multiseed.json'
SECTIONS = sorted(Path('paper/sections').glob('*.tex'))
DATASETS = ['e_ner', 'indian_legal', 'lener_br', 'german_ler_coarse']

fails = []


def check(label, ok, detail=''):
    print(f"  {'OK ' if ok else 'FAIL'} {label}" + (f'  ({detail})' if detail else ''))
    if not ok:
        fails.append(label)


def parse_pm_row(tex, row_label, n_cells):
    """Extrait les (mean, std) d'une ligne de table 'label & $a_{\\pm b}$ & ...'."""
    m = re.search(re.escape(row_label) + r'([^\\]*(?:\\mathbf)?.*?)\\\\', tex)
    if not m:
        return None
    cells = re.findall(r'([0-9]+\.[0-9]+)\}?_\{\\pm([0-9]+\.[0-9]+)\}', m.group(1))
    return [(float(a), float(b)) for a, b in cells][:n_cells] if len(cells) >= n_cells else None


def main():
    agg = json.load(open(AGG, encoding='utf-8'))['agg']
    tex04 = Path('paper/sections/04_experiments.tex').read_text(encoding='utf-8')

    def expected(head, ds):
        e = agg[head]['ami'][ds]
        return round(e['mean'], 1), round(e['std'], 1)

    # ---- Passe 1 : cellules des tables vs JSON ----
    print('== Passe 1 : cellules OPENER des tables vs agrégation ==')
    gold_block = tex04.split(r'\label{tab:legal-gold}')[1].split(r'\end{tabular}')[0]
    main_block = tex04.split(r'\label{tab:legal-main}')[1].split(r'\end{tabular}')[0]
    for head, block, row in [('zs_gold', gold_block, 'OPENER-ZS'),
                             ('sup_gold', gold_block, 'OPENER-Sup'),
                             ('zs_e2e', main_block, 'OPENER-ZS'),
                             ('sup_e2e', main_block, 'OPENER-Sup')]:
        cells = parse_pm_row(block, row, 5)
        if cells is None:
            check(f'{head}: ligne {row} parsée', False)
            continue
        for (got_m, got_s), ds in zip(cells, DATASETS + ['__avg__']):
            exp_m, exp_s = expected(head, ds)
            check(f'{head}/{ds}', (got_m, got_s) == (exp_m, exp_s),
                  f'tex={got_m}±{got_s} json={exp_m}±{exp_s}')

    # ---- Passe 2 : chiffres dérivés du texte (arithmétique des cellules affichées) ----
    print('== Passe 2 : chiffres dérivés du texte ==')
    disp = {h: [expected(h, ds)[0] for ds in DATASETS] for h in agg}
    stds = {h: [expected(h, ds)[1] for ds in DATASETS] for h in agg}
    full = ''.join(p.read_text(encoding='utf-8') for p in SECTIONS)

    derived = {
        # moyenne des cellules affichées, arrondie
        'gold Sup mean 61.4': round(sum(disp['sup_gold']) / 4, 1) == 61.4,
        'gold ZS mean 39.5': round(sum(disp['zs_gold']) / 4, 1) == 39.5,
        'e2e Sup avg 35.9': round(sum(disp['sup_e2e']) / 4, 1) == 35.9,
        'e2e ZS avg 33.4': round(sum(disp['zs_e2e']) / 4, 1) == 33.4,
        # plages et bornes citées
        'ZS cross-lingue 24--37': (round(disp['zs_gold'][3]) == 24
                                   and round(disp['zs_gold'][2]) == 37),
        'ZS seed-std 3--6': (min(stds['zs_gold']) == 3.0 and max(stds['zs_gold']) == 6.0),
        'Sup gold std <= 1.4': max(stds['sup_gold']) == 1.4,
        'per-seed e2e avgs 35.5--36.2': [round(v, 1) for v in sorted(
            agg['sup_e2e']['ami']['__avg__']['per_seed'].values())] == [35.5, 35.9, 36.2],
        'German fine Sup 61.5±0.7': expected('sup_gold', 'german_ler') == (61.5, 0.7),
        'German fine ZS 22.7±2.1': expected('zs_gold', 'german_ler') == (22.7, 2.1),
    }
    for label, ok in derived.items():
        check(label, ok)

    # présence des valeurs citées dans le texte (abstract/results/analysis/conclusion)
    for num in ['70.9', '49.1', '61.4', '47.7', '39.5', '35.9', '33.4',
                '53.7', '60.3', '37.4', '23.8', '61.5', '22.7', '24$--$37']:
        check(f'texte contient {num}', num in full)
    # anciens chiffres qui ne doivent PLUS apparaître comme valeurs OPENER
    for old in ['$70.0$', '$46.3$', '$61.1$', '$44.0$', '$36.1$ mean', '$62.0$', '$22.1$',
                '$22$--$32$', '$35.4$']:
        check(f'ancien chiffre absent : {old}', old not in full)

    # ---- Passe 3 : phrases contenant « seed » (relecture humaine) ----
    print('== Passe 3 : phrases contenant "seed" (à relire) ==')
    for p in SECTIONS:
        for i, line in enumerate(p.read_text(encoding='utf-8').splitlines(), 1):
            for sent in re.split(r'(?<=[.!?])\s+', line):
                if 'seed' in sent.lower():
                    print(f'  {p.name}:{i}: {sent.strip()[:160]}')

    print('\nVERDICT :', 'PASS' if not fails else f'{len(fails)} FAIL : {fails}')
    raise SystemExit(0 if not fails else 1)


if __name__ == '__main__':
    main()
