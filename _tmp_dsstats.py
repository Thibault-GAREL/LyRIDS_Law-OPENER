import collections
from src.data.legal_datasets import load_legal_dataset

for key in ['e_ner', 'indian_legal', 'lener_br', 'german_ler_coarse', 'german_ler']:
    recs = load_legal_dataset(key, split='test')
    n_sent = len(recs)
    n_ment = sum(len(sp) for _, sp in recs)
    c = collections.Counter(t for _, sp in recs for (_, _, t) in sp)
    tot = sum(c.values())
    print(f'== {key}: sent={n_sent} ment={n_ment} types={len(c)}')
    print('   ', ', '.join(f'{k} {100*v/tot:.1f}' for k, v in c.most_common()))
    sub = recs[:1000]
    print(f'    first1000: sent={len(sub)} ment={sum(len(sp) for _, sp in sub)}')
