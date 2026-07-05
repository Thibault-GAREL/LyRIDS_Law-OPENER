"""Bootstrap CIs pour l'AMI END-TO-END (aucun re-entrainement).

Pendant e2e du bootstrap gold (run_bootstrap_ci.py). Ici l'inference e2e est
aussi deterministe (detecteur GLiNER gele + embedder gele) : la seule incertitude
est l'echantillon de test. On rejoue la detection/typing UNE fois par systeme,
on garde les paires alignees (gold, pred) PAR PHRASE, puis on **reechantillonne
les phrases** (bootstrap, K tirages) pour reporter AMI moyen +- std + IC 95%.

Systemes (au seuil 0.3, comme le papier) :
  - GLiNER-M   : labels propres du detecteur GLiNER-M          (baseline)
  - OPENER-Sup : GLiNER-M + LinearSVC balanced fit-on-detected (tete supervisee)
  - GLiNER-L   : labels propres du detecteur GLiNER-L          (baseline)
  - OPENER-ZS  : GLiNER-L + prototypes + transductif + fusion  (tete zero-shot)

Comparaisons APPARIEES : a chaque tirage, on tire les MEMES phrases pour tous les
systemes d'un meme dataset -> l'IC de la difference (ex. OPENER-Sup - GLiNER-L)
dit si l'ecart e2e (35.9 vs 35.0) est distinguable de 0.

Invariance a la graine du tirage : on relance tout avec seeds 42/7/123 et on
verifie que les IC ne bougent pas (K=1000 rend le bootstrap ~independant de la
graine).

Usage :
  python -m scripts.run_bootstrap_e2e --legal --boot 1000 --seeds 42 7 123
"""
import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.metrics import adjusted_mutual_info_score

from src.data.owner_datasets import collect_label_set
from src.models.embedder import Embedder
from src.utils.config import load_config
from scripts.run_opener_e2e import align, fit_classifiers_on_detected
from scripts.run_opener_zs_e2e_fusion import sims_matrix
from scripts.run_opener_zs_sweep import build_prototypes, refine as refine_protos, _norm

SYSTEMS = ['GLiNER-M', 'OPENER-Sup', 'GLiNER-L', 'OPENER-ZS']
PAIRS = [('OPENER-Sup', 'GLiNER-L'), ('OPENER-Sup', 'GLiNER-M'),
         ('OPENER-ZS', 'GLiNER-L'), ('OPENER-Sup', 'OPENER-ZS')]


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def ami(pairs_subset):
    """AMI sur une liste de phrases [(yg_list, yp_list), ...] concatenees."""
    yg, yp = [], []
    for g, p in pairs_subset:
        yg.extend(g); yp.extend(p)
    if not yg:
        return 0.0
    return float(adjusted_mutual_info_score(yg, yp))


def per_sentence_pairs_sup(md_model, embedder, svc, test, labels, thr):
    """Retourne, par phrase, les paires alignees (yg, yp) pour GLiNER-M et OPENER-Sup."""
    gm, sup = [], []
    for text, gold_spans in test:
        ents = md_model.predict_entities(text, labels, threshold=thr)
        det = [(e['start'], e['end'], e['label']) for e in ents]
        if det:
            emb = embedder.embed_entities([text[s:e] for (s, e, _) in det],
                                          full_text=text, spans=[(s, e) for (s, e, _) in det])
            y_svc = list(svc.predict(emb))
        else:
            y_svc = []
        gm.append(align(gold_spans, [(s, e, g) for (s, e, g) in det]))
        sup.append(align(gold_spans, [(det[j][0], det[j][1], y_svc[j]) for j in range(len(det))]))
    return gm, sup


def per_sentence_pairs_zs(md_model, embedder, test, name, labels, anchor_dicts, args):
    """Retourne, par phrase, les paires alignees (yg, yp) pour GLiNER-L et OPENER-ZS (fusion)."""
    labels_order, protos0 = build_prototypes(embedder, name, labels, anchor_dicts,
                                             args.anchor_mode, args.proto_mode)
    per_sentence, allX = [], []
    for text, gold_spans in test:
        ents = md_model.predict_entities(text, labels, threshold=args.threshold)
        det = [(e['start'], e['end'], e['label'], float(e.get('score', 1.0))) for e in ents]
        emb = None
        if det:
            emb = _norm(embedder.embed_entities([text[s:e] for (s, e, _, _) in det],
                        full_text=text, spans=[(s, e) for (s, e, _, _) in det]))
            allX.append(emb)
        per_sentence.append((gold_spans, det, emb))
    protos_tr = (refine_protos(np.vstack(allX), labels_order, protos0, args.refine_iters)
                 if allX else protos0)

    lab2i = {l: i for i, l in enumerate(labels_order)}
    gl, zs = [], []
    for gold_spans, det, emb in per_sentence:
        gl.append(align(gold_spans, [(s, e, g) for (s, e, g, _) in det]))
        if emb is not None and len(det):
            S = sims_matrix(emb, labels_order, protos_tr)
            for j, (_, _, g, sc) in enumerate(det):
                if g in lab2i:
                    S[j, lab2i[g]] += args.beta * sc
            labs = [labels_order[k] for k in S.argmax(axis=1)]
            zs.append(align(gold_spans, [(det[j][0], det[j][1], labs[j]) for j in range(len(det))]))
        else:
            zs.append(align(gold_spans, []))
    return gl, zs


def bootstrap_dataset(pairs_by_system, k, seed):
    """Bootstrap apparie (memes phrases pour tous les systemes). Retourne les
    tableaux AMI (k,) par systeme + le point estimate."""
    n = len(next(iter(pairs_by_system.values())))
    rng = np.random.default_rng(seed)
    point = {s: ami(pairs_by_system[s]) for s in pairs_by_system}
    boot = {s: np.empty(k) for s in pairs_by_system}
    for it in range(k):
        idx = rng.integers(0, n, n)
        for s, pairs in pairs_by_system.items():
            sub = [pairs[i] for i in idx]
            boot[s][it] = ami(sub)
    return point, boot, idx  # idx unused; kept for clarity


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', nargs='+',
                    default=['e_ner', 'indian_legal', 'lener_br', 'german_ler_coarse'])
    ap.add_argument('--legal', action='store_true')
    ap.add_argument('--embedder', default='../LyRIDS_Opener/outputs/models/embedder_contrastive_hard_big')
    ap.add_argument('--gliner-sup', default='urchade/gliner_medium-v2.1')
    ap.add_argument('--gliner-zs', default='urchade/gliner_large-v2.1')
    ap.add_argument('--threshold', type=float, default=0.3)
    ap.add_argument('--beta', type=float, default=0.05)
    ap.add_argument('--proto-mode', default='ensemble')
    ap.add_argument('--refine-iters', type=int, default=3)
    ap.add_argument('--anchor-mode', choices=['dict', 'auto'], default='dict')
    ap.add_argument('--anchor-dict', default='configs/legal_anchors.yaml')
    ap.add_argument('--task-prefix', default='classification: ')
    ap.add_argument('--max-train', type=int, default=2000)
    ap.add_argument('--max-eval', type=int, default=1000)
    ap.add_argument('--boot', type=int, default=1000)
    ap.add_argument('--seeds', nargs='+', type=int, default=[42, 7, 123])
    ap.add_argument('--output-dir', default='outputs/results/legal/bootstrap_e2e')
    ap.add_argument('--resume', action='store_true', help='Reprend depuis le checkpoint Phase A.')
    args = ap.parse_args()

    if args.legal:
        from src.data.legal_datasets import load_legal_dataset
        load_ds = load_legal_dataset
    else:
        from src.data.owner_datasets import load_owner_dataset as load_ds

    import torch
    from gliner import GLiNER
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    log(f"=== Bootstrap e2e (K={args.boot}, seeds={args.seeds}) sur {device} ===")
    log(f"Embedder {args.embedder} | thr={args.threshold} | datasets={args.datasets}")
    embedder = Embedder(model_name=args.embedder, truncate_dim=None,
                        encoding_mode='span_in_context', task_prefix=args.task_prefix)
    anchor_dicts = load_config(args.anchor_dict) or {} if args.anchor_mode == 'dict' else {}

    def _load(name, split, cap):
        try:
            return load_ds(name, split=split, max_sentences=cap)
        except Exception:
            return load_ds(name, split='validation', max_sentences=cap)

    # pairs_all[dataset][system] = liste de (yg, yp) par phrase
    pairs_all = {}
    labels_by_ds = {}
    for name in args.datasets:
        test = _load(name, 'test', args.max_eval)
        labels_by_ds[name] = (collect_label_set(test), len(test))
        pairs_all[name] = {}

    # --- Phase A : detecteur GLiNER-M (OPENER-Sup + baseline GLiNER-M) ---
    ckpt = Path(args.output_dir) / 'phaseA_cache.json'
    if args.resume and ckpt.exists():
        log(f"Reprise : Phase A chargee depuis {ckpt}")
        cached = json.loads(ckpt.read_text(encoding='utf-8'))
        for name in args.datasets:
            for s in ('GLiNER-M', 'OPENER-Sup'):
                pairs_all[name][s] = [tuple(x) for x in cached[name][s]]
    else:
        log(f"Chargement detecteur SUP {args.gliner_sup} ...")
        md = GLiNER.from_pretrained(args.gliner_sup)
        if device == 'cuda':
            md = md.to(device)
        for name in args.datasets:
            labels, ntest = labels_by_ds[name]
            train = _load(name, 'train', args.max_train)
            test = _load(name, 'test', args.max_eval)
            log(f"[GLiNER-M] {name}: fit-on-detected (train={len(train)}) ...")
            fitted, ntr, ncl = fit_classifiers_on_detected(md, embedder, train, labels, args.threshold)
            svc = fitted['linear_svm_balanced']
            log(f"[GLiNER-M] {name}: predictions e2e (test={ntest}) ...")
            gm, sup = per_sentence_pairs_sup(md, embedder, svc, test, labels, args.threshold)
            pairs_all[name]['GLiNER-M'] = gm
            pairs_all[name]['OPENER-Sup'] = sup
        del md
        if device == 'cuda':
            torch.cuda.empty_cache()
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        ckpt.write_text(json.dumps({n: {s: pairs_all[n][s] for s in ('GLiNER-M', 'OPENER-Sup')}
                                    for n in args.datasets}, ensure_ascii=False), encoding='utf-8')
        log(f"Checkpoint Phase A -> {ckpt}")

    # --- Phase B : detecteur GLiNER-L (OPENER-ZS + baseline GLiNER-L) ---
    log(f"Chargement detecteur ZS {args.gliner_zs} ...")
    md = GLiNER.from_pretrained(args.gliner_zs)
    if device == 'cuda':
        md = md.to(device)
    for name in args.datasets:
        labels, ntest = labels_by_ds[name]
        test = _load(name, 'test', args.max_eval)
        log(f"[GLiNER-L] {name}: prototypes + transductif + fusion (test={ntest}) ...")
        gl, zs = per_sentence_pairs_zs(md, embedder, test, name, labels, anchor_dicts, args)
        pairs_all[name]['GLiNER-L'] = gl
        pairs_all[name]['OPENER-ZS'] = zs
    del md
    if device == 'cuda':
        torch.cuda.empty_cache()

    # --- Bootstrap (par dataset + Avg apparie), pour chaque graine ---
    log("Rejeu termine. Bootstrap CPU ...")
    out = {'params': vars(args), 'per_dataset': {}, 'avg': {}, 'seeds': {}}

    # Point estimates par dataset + Avg
    point_ds = {name: {s: ami(pairs_all[name][s]) for s in SYSTEMS} for name in args.datasets}
    out['point'] = {'per_dataset': point_ds,
                    'avg': {s: float(np.mean([point_ds[d][s] for d in args.datasets])) for s in SYSTEMS}}

    for seed in args.seeds:
        # Per-dataset : tableaux AMI (K,) par systeme, tirage apparie intra-dataset
        boot_ds = {}
        for di, name in enumerate(args.datasets):
            n = labels_by_ds[name][1]
            rng = np.random.default_rng(seed + 1000 * di)
            arr = {s: np.empty(args.boot) for s in SYSTEMS}
            idx_store = rng.integers(0, n, size=(args.boot, n))
            for it in range(args.boot):
                idx = idx_store[it]
                for s in SYSTEMS:
                    arr[s][it] = ami([pairs_all[name][s][i] for i in idx])
            boot_ds[name] = arr
        # Avg apparie : moyenne des 4 datasets par tirage (memes idx intra-dataset)
        avg_arr = {s: np.mean([boot_ds[d][s] for d in args.datasets], axis=0) for s in SYSTEMS}
        diff_arr = {f"{a}-{b}": avg_arr[a] - avg_arr[b] for a, b in PAIRS}

        def _stat(a):
            a = np.asarray(a, float) * 100
            return {'mean': float(a.mean()), 'std': float(a.std(ddof=1)),
                    'ci95': [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]}

        out['seeds'][str(seed)] = {
            'avg': {s: _stat(avg_arr[s]) for s in SYSTEMS},
            'avg_diff': {k: _stat(v) for k, v in diff_arr.items()},
            'per_dataset': {name: {s: _stat(boot_ds[name][s]) for s in SYSTEMS}
                            for name in args.datasets},
        }
        d = out['seeds'][str(seed)]
        log(f"--- seed {seed} : Avg AMI ---")
        for s in SYSTEMS:
            st = d['avg'][s]
            log(f"    {s:11} {st['mean']:5.1f} +- {st['std']:.2f}  CI[{st['ci95'][0]:.1f},{st['ci95'][1]:.1f}]")
        log(f"--- seed {seed} : differences appariees (Avg) ---")
        for k, st in d['avg_diff'].items():
            sig = "SIGNIF" if (st['ci95'][0] > 0 or st['ci95'][1] < 0) else "n.s. (CI contient 0)"
            log(f"    {k:23} {st['mean']:+5.2f}  CI[{st['ci95'][0]:+.2f},{st['ci95'][1]:+.2f}]  {sig}")

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    ds = datetime.now().strftime('%Y-%m-%d_%H%M%S')
    fp = Path(args.output_dir) / f'bootstrap_e2e_{ds}.json'
    fp.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding='utf-8')
    log(f"Sauvegarde -> {fp}")


if __name__ == '__main__':
    main()
