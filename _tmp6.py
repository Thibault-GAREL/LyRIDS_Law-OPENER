import json

import yaml

j = json.load(open('outputs/results/bootstrap_ci/SUMMARY_bootstrap_ci_2026-07-05_234213.json',
                   encoding='utf-8'))
for ds, r in j['results'].items():
    b = r['bootstrap']
    print(ds,
          'sup std=%.1f' % (100 * b['sup']['ami']['std']),
          'zs std=%.1f' % (100 * b['zs']['ami']['std']),
          '| point sup=%.1f zs=%.1f' % (100 * r['point']['sup']['ami'], 100 * r['point']['zs']['ami']),
          '| macroF1 sup=%.1f zs=%.1f' % (100 * r['point']['sup']['macro_f1'],
                                          100 * r['point']['zs']['macro_f1']))

anch = yaml.safe_load(open('configs/legal_anchors.yaml', encoding='utf-8'))
print('\nIndian anchors:')
for t, w in anch['indian_legal'].items():
    print(' ', t, w)
