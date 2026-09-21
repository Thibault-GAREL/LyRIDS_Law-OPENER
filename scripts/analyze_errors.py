"""Analyses camera-ready calculees sur les dumps de scripts/dump_predictions.py.

Tout est fait hors GPU, a partir des spans sauvegardes, de sorte qu'on peut
iterer sur les analyses sans relancer la moindre inference.

Produit les trois choses demandees par les reviewers NLLP 2026 :

 1. F1 entity-level (reviewer qviX) : micro precision / rappel / F1 au sens
    CoNLL (offsets exacts ET type correct), la metrique que rapportent E-NER,
    LeNER-Br, German-LER et Indian-Legal, donc directement comparable a la
    litterature. Le macro-F1 des scripts existants, lui, compte les sentinelles
    FP/FN comme des classes et n'est comparable a rien.

 2. Plafond oracle (reviewer bkDi) : l'AMI et le F1 qu'atteindrait un systeme
    au typage PARFAIT sur exactement les spans que le detecteur a proposes.
    Repond a "quelle est la valeur maximale atteignable" sans faire tourner un
    LLM frontier, et chiffre la these du papier (la detection est le plafond).

 3. Analyse qualitative (reviewer bkDi) : typologie des erreurs de detection
    (frontiere vs mention totalement manquee vs span parasite), matrices de
    confusion de typage sur les mentions correctement detectees, et exemples
    reels extraits du corpus pour le papier.

Usage :
  python -m scripts.analyze_errors
  python -m scripts.analyze_errors --examples-for OPENER-ZS OPENER-Sup
"""
import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

from sklearn.metrics import adjusted_mutual_info_score

LABEL_FN = '__gold_not_predicted__'
LABEL_FP = '__predicted_not_gold__'
# Ordre d'affichage. Les systemes absents d'un dump sont simplement sautes, et
# une phrase qu'un systeme n'a pas evaluee (Qwen, limite a 200) est ignoree
# pour CE systeme seulement, au lieu de compter comme une phrase sans
# prediction, ce qui le penaliserait a tort.
SYSTEMS = ['GLiNER-S', 'GLiNER-M', 'GLiNER-L', 'GNER-T5', 'Qwen-1.5B',
           'OPENER-Sup', 'OPENER-ZS']


def _evaluated(sentences, system):
    """Phrases sur lesquelles ce systeme a reellement tourne."""
    return [s for s in sentences if system in s['systems']]


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


# --------------------------------------------------------------------------
# Metriques
# --------------------------------------------------------------------------
def ami_with_sentinels(sentences, system, oracle=False):
    """AMI du papier : alignement par offsets exacts + sentinelles FP/FN.

    oracle=True donne le PLAFOND atteignable sur les spans que ce detecteur a
    proposes : chaque span correctement detecte recoit son type gold, et tous
    les spans parasites recoivent une seule et meme etiquette.

    Regrouper les parasites est indispensable. Si on leur laissait le type que
    le systeme leur avait donne, la borne dependrait du systeme qu'elle est
    censee borner, et deux systemes partageant un detecteur (donc des spans
    identiques) auraient des plafonds differents. Cette borne suppose en
    revanche qu'un oracle reconnaisse un span parasite, ce qui releve de la
    detection et non du typage : c'est donc une borne superieure large.
    """
    y_gold, y_pred = [], []
    for s in _evaluated(sentences, system):
        gold_d = {(a, b): lbl for a, b, lbl, _ in s['gold']}
        pred_d = {(a, b): lbl for a, b, lbl, _ in s['systems'][system]}
        for k in set(gold_d) | set(pred_d):
            g = gold_d.get(k, LABEL_FP)
            p = pred_d.get(k, LABEL_FN)
            if oracle and k in pred_d:
                p = g if k in gold_d else LABEL_FP
            y_gold.append(g)
            y_pred.append(p)
    if not y_gold:
        return 0.0
    return float(adjusted_mutual_info_score(y_gold, y_pred))


def entity_level_prf(sentences, system, typed=True):
    """Micro P/R/F1 entity-level (CoNLL).

    typed=True  : un span compte comme correct si offsets ET type sont bons.
    typed=False : detection seule (offsets), c'est-a-dire le plafond oracle.
    """
    tp = n_pred = n_gold = 0
    for s in _evaluated(sentences, system):
        gold_d = {(a, b): lbl for a, b, lbl, _ in s['gold']}
        pred_d = {(a, b): lbl for a, b, lbl, _ in s['systems'][system]}
        n_gold += len(gold_d)
        n_pred += len(pred_d)
        for k, p in pred_d.items():
            if k in gold_d and (not typed or gold_d[k] == p):
                tp += 1
    prec = tp / n_pred if n_pred else 0.0
    rec = tp / n_gold if n_gold else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {'precision': prec, 'recall': rec, 'f1': f1,
            'tp': tp, 'n_pred': n_pred, 'n_gold': n_gold}


def typing_accuracy_on_detected(sentences, system):
    """Justesse du typage sur les seules mentions correctement detectees.

    Isole la tete de typage de la detection : c'est le chiffre qui dit si le
    goulot est la detection (accuracy haute) ou le typage (accuracy basse).
    """
    ok = tot = 0
    for s in _evaluated(sentences, system):
        gold_d = {(a, b): lbl for a, b, lbl, _ in s['gold']}
        for a, b, lbl, _ in s['systems'][system]:
            if (a, b) in gold_d:
                tot += 1
                ok += (gold_d[(a, b)] == lbl)
    return {'accuracy': ok / tot if tot else 0.0, 'n': tot}


def detection_breakdown(sentences, system):
    """Typologie des erreurs de detection, cote gold et cote predictions.

    Un span gold est : exact (offsets retrouves), boundary (chevauche par une
    prediction mais bornes differentes) ou missed (aucun chevauchement). Une
    prediction est : exact, boundary, ou spurious (aucun gold chevauche).
    """
    g_exact = g_boundary = g_missed = 0
    p_exact = p_boundary = p_spurious = 0
    for s in _evaluated(sentences, system):
        gold = [(a, b) for a, b, _, _ in s['gold']]
        pred = [(a, b) for a, b, _, _ in s['systems'][system]]
        gset, pset = set(gold), set(pred)
        for (a, b) in gold:
            if (a, b) in pset:
                g_exact += 1
            elif any(a < d and c < b for (c, d) in pred):
                g_boundary += 1
            else:
                g_missed += 1
        for (a, b) in pred:
            if (a, b) in gset:
                p_exact += 1
            elif any(a < d and c < b for (c, d) in gold):
                p_boundary += 1
            else:
                p_spurious += 1
    n_g = max(g_exact + g_boundary + g_missed, 1)
    n_p = max(p_exact + p_boundary + p_spurious, 1)
    return {
        'gold': {'exact': g_exact, 'boundary': g_boundary, 'missed': g_missed,
                 'exact_pct': 100 * g_exact / n_g, 'boundary_pct': 100 * g_boundary / n_g,
                 'missed_pct': 100 * g_missed / n_g},
        'pred': {'exact': p_exact, 'boundary': p_boundary, 'spurious': p_spurious,
                 'exact_pct': 100 * p_exact / n_p, 'boundary_pct': 100 * p_boundary / n_p,
                 'spurious_pct': 100 * p_spurious / n_p},
    }


def confusions(sentences, system, top=12):
    """Confusions de typage (gold -> predit) sur les mentions bien detectees."""
    c = Counter()
    for s in _evaluated(sentences, system):
        gold_d = {(a, b): lbl for a, b, lbl, _ in s['gold']}
        for a, b, lbl, _ in s['systems'][system]:
            if (a, b) in gold_d and gold_d[(a, b)] != lbl:
                c[(gold_d[(a, b)], lbl)] += 1
    return [{'gold': g, 'pred': p, 'n': n} for (g, p), n in c.most_common(top)]


def error_examples(sentences, systems, limit=40):
    """Exemples reels pour le papier : mentions bien detectees ou les deux tetes
    divergent, ou l'une des deux se trompe. Ce sont les cas qui illustrent la
    difference zero-shot / supervise demandee par le reviewer bkDi.
    """
    out = []
    for s in sentences:
        gold_d = {(a, b): lbl for a, b, lbl, _ in s['gold']}
        preds = {sys: {(a, b): lbl for a, b, lbl, _ in s['systems'].get(sys, [])}
                 for sys in systems}
        surface = {(a, b): txt for a, b, _, txt in s['gold']}
        for k, g in gold_d.items():
            got = {sys: preds[sys].get(k) for sys in systems}
            if any(v is None for v in got.values()):
                continue
            if all(v == g for v in got.values()):
                continue
            out.append({'text': s['text'][:220], 'mention': surface[k],
                        'gold': g, **{f'pred_{sys}': got[sys] for sys in systems}})
            if len(out) >= limit:
                return out
    return out


def missed_examples(sentences, system, limit=25):
    """Mentions gold que le detecteur rate completement ou dont il rate les
    bornes. Illustre que le plafond end-to-end est fixe par la detection.
    """
    out = []
    for s in sentences:
        pred = [(a, b) for a, b, _, _ in s['systems'].get(system, [])]
        pset = set(pred)
        for a, b, lbl, txt in s['gold']:
            if (a, b) in pset:
                continue
            overlap = [(c, d) for (c, d) in pred if a < d and c < b]
            out.append({'mention': txt, 'gold': lbl, 'len_chars': b - a,
                        'kind': 'boundary' if overlap else 'missed',
                        'pred_span': (s['text'][overlap[0][0]:overlap[0][1]]
                                      if overlap else None)})
            if len(out) >= limit:
                return out
    return out


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dump-dir', default='outputs/results/legal/error_analysis')
    ap.add_argument('--datasets', nargs='+',
                    default=['e_ner', 'indian_legal', 'lener_br', 'german_ler_coarse'])
    ap.add_argument('--examples-for', nargs='+', default=['OPENER-ZS', 'OPENER-Sup'])
    args = ap.parse_args()

    dump_dir = Path(args.dump_dir)
    report = {'generated': datetime.now().isoformat(timespec='seconds'), 'datasets': {}}

    for name in args.datasets:
        fp = dump_dir / f'dump_{name}.json'
        if not fp.exists():
            log(f"!! dump manquant : {fp}")
            continue
        d = json.loads(fp.read_text(encoding='utf-8'))
        sents = d['sentences']
        log(f"=== {name} ({len(sents)} phrases, {sum(len(s['gold']) for s in sents)} mentions) ===")
        entry = {'n_sentences': len(sents),
                 'n_gold_mentions': sum(len(s['gold']) for s in sents),
                 'labels': d['labels'], 'systems': {}}

        for sys in SYSTEMS:
            n_eval = len(_evaluated(sents, sys))
            if n_eval == 0:
                continue                      # systeme absent de ce dump
            m = {
                'n_sentences_evaluated': n_eval,
                'ami': ami_with_sentinels(sents, sys),
                'ami_oracle': ami_with_sentinels(sents, sys, oracle=True),
                'prf_typed': entity_level_prf(sents, sys, typed=True),
                'prf_detection': entity_level_prf(sents, sys, typed=False),
                'typing_acc_on_detected': typing_accuracy_on_detected(sents, sys),
                'detection_breakdown': detection_breakdown(sents, sys),
                'confusions': confusions(sents, sys),
            }
            entry['systems'][sys] = m
            p, r, f = (m['prf_typed'][k] for k in ('precision', 'recall', 'f1'))
            log(f"  {sys:11} n={n_eval:4}  AMI {100*m['ami']:5.1f}"
                f"  (oracle {100*m['ami_oracle']:5.1f})"
                f"   P {100*p:5.1f}  R {100*r:5.1f}  F1 {100*f:5.1f}"
                f"   typing-acc {100*m['typing_acc_on_detected']['accuracy']:5.1f}")

        entry['examples_divergent'] = error_examples(sents, args.examples_for)
        entry['examples_missed_glinerL'] = missed_examples(sents, 'GLiNER-L')
        report['datasets'][name] = entry

    outp = dump_dir / 'ANALYSIS_errors.json'
    outp.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    log(f"Rapport -> {outp}")


if __name__ == '__main__':
    main()
