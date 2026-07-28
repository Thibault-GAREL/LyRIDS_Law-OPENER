import collections
import json
import re

from src.data.legal_datasets import load_legal_dataset

# 1. Part des NRM+RS sur les 1000 premieres phrases de German-LER (sous-ensemble evalue)
recs = load_legal_dataset('german_ler_coarse', split='test')[:1000]
c = collections.Counter(t for _, sp in recs for (_, _, t) in sp)
tot = sum(c.values())
print('German-LER first1000 shares:', {k: round(100 * v / tot, 1) for k, v in c.most_common()})
print('  NRM+RS =', round(100 * (c['NRM'] + c['RS']) / tot, 1))

# 2. Ancres : nombre de mots par type
import yaml
anch = yaml.safe_load(open('configs/legal_anchors.yaml', encoding='utf-8'))
for ds, types in anch.items():
    if not isinstance(types, dict):
        continue
    lens = {t: len(w) for t, w in types.items()}
    print(f'{ds}: {len(types)} types, mots/type min={min(lens.values())} max={max(lens.values())}')

# 3. bootstrap gold
j = json.load(open('outputs/results/bootstrap_ci/SUMMARY_bootstrap_ci_2026-07-05_234213.json', encoding='utf-8'))
print(json.dumps(j, indent=1)[:1800])
