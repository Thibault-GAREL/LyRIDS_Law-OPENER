"""Latence + energie OWNER par dataset legal, methode du papier generaliste.

OWNER tourne en batch (pas d'appel par phrase) : on prend le wall-clock par
dataset depuis MLflow (meta.yaml start/end du run e2e), puis on amortit comme
`scripts/baselines/owner_collect.py` du journal :
    energie (Wh)   = wall_clock_s * 110 W / 3600      (TDP estimate, +/-20%)
    latence (ms)   = wall_clock_s * 1000 / n_sentences

Les run_id par dataset sont lus depuis le dernier owner_e2e_*.json.

Usage :
    python -m scripts.baselines.owner_legal_efficiency
"""
import glob
import json
import os
import re

MLRUNS = '../LyRIDS_Opener/external/OWNER/mlruns'
DATA = '../LyRIDS_Opener/external/OWNER/data/lyrids'
DATASETS = ['e_ner', 'indian_legal', 'lener_br', 'german_ler_coarse']
POWER_W = 110.0  # meme estimation que owner_collect.py (GPU ~80W + CPU ~30W)


def latest_owner_e2e_runs():
    for f in sorted(glob.glob('outputs/results/baselines/owner/owner_e2e_*.json'),
                    key=os.path.getmtime, reverse=True):
        r = json.load(open(f, encoding='utf-8'))['results']
        if all(d in r for d in DATASETS):
            return {d: r[d]['mlflow_run'] for d in DATASETS}
    raise SystemExit('Aucun owner_e2e_*.json couvrant les 4 datasets legaux.')


def wall_clock_s(run_id):
    hits = glob.glob(os.path.join(MLRUNS, '*', run_id, 'meta.yaml'))
    if not hits:
        return None
    txt = open(hits[0], encoding='utf-8').read()
    st = re.search(r'start_time:\s*(\d+)', txt)
    et = re.search(r'end_time:\s*(\d+)', txt)
    return (int(et.group(1)) - int(st.group(1))) / 1000.0 if st and et else None


def n_sentences(ds):
    return len(json.load(open(os.path.join(DATA, ds, 'test.json'), encoding='utf-8'))['documents'])


def main():
    runs = latest_owner_e2e_runs()
    print(f"{'dataset':18} {'n_sent':>6} {'wall_s':>8} {'lat_ms':>7} {'Wh':>6}")
    lat, wh = [], []
    for ds in DATASETS:
        t = wall_clock_s(runs[ds])
        n = n_sentences(ds)
        if t is None:
            print(f"{ds:18} run {runs[ds][:8]} : timing MLflow introuvable")
            continue
        l = t * 1000 / n
        e = t * POWER_W / 3600
        lat.append(l)
        wh.append(e)
        print(f"{ds:18} {n:6d} {t:8.1f} {l:7.0f} {e:6.1f}")
    if lat:
        print(f"{'AVG (mean)':18} {'':>6} {'':>8} {sum(lat)/len(lat):7.0f} {sum(wh)/len(wh):6.1f}")


if __name__ == '__main__':
    main()
