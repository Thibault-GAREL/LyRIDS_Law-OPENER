"""Dump des predictions end-to-end, span par span, avec le TEXTE des mentions.

Motivation (camera-ready NLLP 2026, reviewers qviX / bkDi) : les scripts d'eval
existants ne gardent que des metriques agregees, ce qui interdit (1) de calculer
un F1 entity-level standard a posteriori, (2) de montrer le moindre exemple reel
d'erreur dans le papier. On rejoue donc l'inference UNE fois par systeme et on
ecrit tout sur disque. Toutes les analyses (F1, oracle, confusions, exemples)
sont ensuite faites hors GPU par scripts/analyze_errors.py.

Systemes rejoues (identiques au papier, seuil 0.3) :
  - GLiNER-M   : labels propres du detecteur                    (baseline)
  - OPENER-Sup : GLiNER-M + LinearSVC balanced fit-on-detected  (tete supervisee)
  - GLiNER-L   : labels propres du detecteur                    (baseline)
  - OPENER-ZS  : GLiNER-L + prototypes + transductif + fusion   (tete zero-shot)

Aucun re-entrainement : detecteur et embedder sont geles, seule la probe
lineaire est fittee sur le train cible, exactement comme dans le papier.

Usage :
  python -m scripts.dump_predictions --legal
"""
import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np

from src.data.owner_datasets import collect_label_set
from src.models.embedder import Embedder
from src.utils.config import load_config
from scripts.run_opener_e2e import fit_classifiers_on_detected
from scripts.run_opener_zs_e2e_fusion import sims_matrix
from scripts.run_opener_zs_sweep import build_prototypes, refine as refine_protos, _norm


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def _spans_with_surface(text, spans):
    """[(start, end, label)] -> [[start, end, label, surface]] pour le dump."""
    return [[int(s), int(e), lbl, text[s:e]] for (s, e, lbl) in spans]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', nargs='+',
                    default=['e_ner', 'indian_legal', 'lener_br', 'german_ler_coarse'])
    ap.add_argument('--legal', action='store_true')
    ap.add_argument('--embedder',
                    default='../LyRIDS_Opener/outputs/models/embedder_contrastive_hard_big')
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
    ap.add_argument('--output-dir', default='outputs/results/legal/error_analysis')
    args = ap.parse_args()

    if args.legal:
        from src.data.legal_datasets import load_legal_dataset as load_ds
    else:
        from src.data.owner_datasets import load_owner_dataset as load_ds

    import torch
    from gliner import GLiNER
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    log(f"=== Dump des predictions e2e sur {device} ===")
    log(f"Embedder {args.embedder} | thr={args.threshold} | beta={args.beta}")
    log(f"Datasets : {args.datasets}")
    embedder = Embedder(model_name=args.embedder, truncate_dim=None,
                        encoding_mode='span_in_context', task_prefix=args.task_prefix)
    anchor_dicts = load_config(args.anchor_dict) or {} if args.anchor_mode == 'dict' else {}

    def _load(name, split, cap):
        try:
            return load_ds(name, split=split, max_sentences=cap)
        except Exception:
            return load_ds(name, split='validation', max_sentences=cap)

    # dump[dataset] = {'sentences': [{text, gold, systems:{...}}], 'labels': [...]}
    dump = {}
    for name in args.datasets:
        test = _load(name, 'test', args.max_eval)
        dump[name] = {
            'labels': sorted(collect_label_set(test)),
            'sentences': [{'text': t, 'gold': _spans_with_surface(t, g), 'systems': {}}
                          for t, g in test],
        }
        log(f"  {name}: {len(test)} phrases, {len(dump[name]['labels'])} types")

    # --- Phase A : detecteur GLiNER-M -> baseline GLiNER-M + tete OPENER-Sup ---
    log(f"Chargement detecteur SUP {args.gliner_sup} ...")
    md = GLiNER.from_pretrained(args.gliner_sup)
    if device == 'cuda':
        md = md.to(device)

    for name in args.datasets:
        labels = dump[name]['labels']
        train = _load(name, 'train', args.max_train)
        test = _load(name, 'test', args.max_eval)
        log(f"[GLiNER-M] {name}: fit-on-detected (train={len(train)}) ...")
        fitted, ntr, ncl = fit_classifiers_on_detected(md, embedder, train, labels, args.threshold)
        svc = fitted['linear_svm_balanced']
        log(f"[GLiNER-M] {name}: inference e2e sur {len(test)} phrases ...")
        for i, (text, _gold) in enumerate(test):
            ents = md.predict_entities(text, labels, threshold=args.threshold)
            det = [(e['start'], e['end'], e['label']) for e in ents]
            if det:
                emb = embedder.embed_entities([text[s:e] for (s, e, _) in det],
                                              full_text=text,
                                              spans=[(s, e) for (s, e, _) in det])
                y_svc = list(svc.predict(emb))
            else:
                y_svc = []
            rec = dump[name]['sentences'][i]['systems']
            rec['GLiNER-M'] = _spans_with_surface(text, det)
            rec['OPENER-Sup'] = _spans_with_surface(
                text, [(det[j][0], det[j][1], y_svc[j]) for j in range(len(det))])
            if (i + 1) % 200 == 0:
                log(f"    ... {i + 1}/{len(test)}")
    del md
    if device == 'cuda':
        torch.cuda.empty_cache()

    # --- Phase B : detecteur GLiNER-L -> baseline GLiNER-L + tete OPENER-ZS ---
    log(f"Chargement detecteur ZS {args.gliner_zs} ...")
    md = GLiNER.from_pretrained(args.gliner_zs)
    if device == 'cuda':
        md = md.to(device)

    for name in args.datasets:
        labels = dump[name]['labels']
        test = _load(name, 'test', args.max_eval)
        log(f"[GLiNER-L] {name}: prototypes + transductif + fusion ({len(test)} phrases) ...")
        labels_order, protos0 = build_prototypes(embedder, name, labels, anchor_dicts,
                                                 args.anchor_mode, args.proto_mode)
        # Passe 1 : detection + embedding (le transductif a besoin de tout le test).
        per_sentence, allX = [], []
        for text, _gold in test:
            ents = md.predict_entities(text, labels, threshold=args.threshold)
            det = [(e['start'], e['end'], e['label'], float(e.get('score', 1.0))) for e in ents]
            emb = None
            if det:
                emb = _norm(embedder.embed_entities([text[s:e] for (s, e, _, _) in det],
                            full_text=text, spans=[(s, e) for (s, e, _, _) in det]))
                allX.append(emb)
            per_sentence.append((text, det, emb))
        protos_tr = (refine_protos(np.vstack(allX), labels_order, protos0, args.refine_iters)
                     if allX else protos0)

        # Passe 2 : typage par prototypes raffines + fusion detecteur.
        lab2i = {l: i for i, l in enumerate(labels_order)}
        for i, (text, det, emb) in enumerate(per_sentence):
            rec = dump[name]['sentences'][i]['systems']
            rec['GLiNER-L'] = _spans_with_surface(text, [(s, e, g) for (s, e, g, _) in det])
            if emb is not None and len(det):
                S = sims_matrix(emb, labels_order, protos_tr)
                for j, (_, _, g, sc) in enumerate(det):
                    if g in lab2i:
                        S[j, lab2i[g]] += args.beta * sc
                labs = [labels_order[k] for k in S.argmax(axis=1)]
                rec['OPENER-ZS'] = _spans_with_surface(
                    text, [(det[j][0], det[j][1], labs[j]) for j in range(len(det))])
            else:
                rec['OPENER-ZS'] = []
    del md
    if device == 'cuda':
        torch.cuda.empty_cache()

    # --- Ecriture : un fichier par dataset (les dumps sont volumineux) ---
    stamp = datetime.now().strftime('%Y-%m-%d_%H%M%S')
    for name in args.datasets:
        fp = outdir / f'dump_{name}.json'
        fp.write_text(json.dumps({'params': vars(args), 'stamp': stamp, **dump[name]},
                                 ensure_ascii=False), encoding='utf-8')
        n_sent = len(dump[name]['sentences'])
        n_gold = sum(len(s['gold']) for s in dump[name]['sentences'])
        log(f"Sauvegarde {fp}  ({n_sent} phrases, {n_gold} mentions gold)")
    log("Termine. Analyses : python -m scripts.analyze_errors")


if __name__ == '__main__':
    main()
