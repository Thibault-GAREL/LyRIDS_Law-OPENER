import json, glob
DS = ['e_ner','indian_legal','lener_br','german_ler_coarse','german_ler']
for tag, pat in [('sup_gold','outputs/results/legal/sup_gold/SUMMARY_*.json'),
                 ('zs_gold','outputs/results/legal/zs_gold/SUMMARY_opener_zs_dict_*.json')]:
    print('###', tag)
    merged={}
    for f in sorted(glob.glob(pat)):
        j=json.load(open(f,encoding='utf-8'))
        print('  params:', json.dumps(j.get('params'))[:400])
        for k,v in (j.get('results') or {}).items():
            if k in DS: merged.setdefault(k,v)
        break
    for k in DS:
        if k in merged:
            print(' ', k, json.dumps(merged[k])[:400])
