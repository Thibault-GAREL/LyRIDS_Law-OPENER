"""Ajoute les predictions des baselines aux dumps de scripts/dump_predictions.py.

But (camera-ready NLLP 2026) : la table de decomposition detection/typage ne
couvrait que les 4 systemes du pipeline OPENER, ce qui pouvait se lire comme une
selection avantageuse. Ce script rejoue GLiNER-S, GNER-T5 et Qwen2.5-1.5B avec
EXACTEMENT les parametres du benchmark (scripts/run_legal_baselines.sh) et
ecrit leurs spans dans les memes fichiers dump_{dataset}.json.

OWNER n'est pas traite ici : sa codebase vit dans l'environnement conda du repo
principal (voir scripts/run_legal_owner.sh) et ses clusters sont nommes
dynamiquement, donc ni justesse de typage ni F1 ne sont definis pour lui.

Une phrase qu'un systeme n'a pas evaluee (cas de Qwen, limite a 200 phrases)
n'a PAS de cle dans 'systems', et analyze_errors l'ignore alors pour ce
systeme au lieu de la compter comme une phrase sans prediction.

Usage :
  python -m scripts.dump_baselines --legal --systems gliner_small qwen1_5b gner
"""
import argparse
import json
from datetime import datetime
from pathlib import Path

# Noms de systeme tels qu'ils apparaitront dans les dumps et les tables.
SYS_NAMES = {
    'gliner_small': 'GLiNER-S',
    'gner': 'GNER-T5',
    'qwen1_5b': 'Qwen-1.5B',
}


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def _spans_with_surface(text, spans):
    return [[int(s), int(e), lbl, text[s:e]] for (s, e, lbl) in spans]


def load_dumps(dump_dir, datasets):
    dumps = {}
    for name in datasets:
        fp = Path(dump_dir) / f'dump_{name}.json'
        if not fp.exists():
            raise SystemExit(f"dump manquant : {fp}. Lancer d'abord "
                             f"python -m scripts.dump_predictions --legal")
        dumps[name] = json.loads(fp.read_text(encoding='utf-8'))
        log(f"  {name}: {len(dumps[name]['sentences'])} phrases chargees")
    return dumps


def save_dumps(dump_dir, dumps):
    for name, d in dumps.items():
        fp = Path(dump_dir) / f'dump_{name}.json'
        fp.write_text(json.dumps(d, ensure_ascii=False), encoding='utf-8')
        log(f"  ecrit {fp}")


# --------------------------------------------------------------------------
def run_gliner(dumps, checkpoint, threshold, device):
    """GLiNER-S : memes parametres que scripts/run_legal_baselines.sh."""
    from gliner import GLiNER
    from scripts.baselines.run_gliner import gliner_predict
    log(f"Chargement {checkpoint} ...")
    model = GLiNER.from_pretrained(checkpoint)
    if device == 'cuda':
        model = model.to(device)
    for name, d in dumps.items():
        labels = d['labels']
        log(f"[GLiNER-S] {name}: {len(d['sentences'])} phrases ...")
        for i, s in enumerate(d['sentences']):
            spans = gliner_predict(model, s['text'], labels, threshold)
            s['systems']['GLiNER-S'] = _spans_with_surface(s['text'], spans)
            if (i + 1) % 250 == 0:
                log(f"    ... {i + 1}/{len(d['sentences'])}")
    del model


def run_gner(dumps, checkpoint, device, batch_size, max_new_tokens,
             max_input_length):
    """GNER-T5-base, prediction batchee comme dans run_gner.py."""
    import torch
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    from scripts.baselines.run_gner import _predict_corpus_batched
    log(f"Chargement {checkpoint} ...")
    dtype = torch.float16 if device == 'cuda' else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint, dtype=dtype)
    model = model.to(device).eval()
    for name, d in dumps.items():
        labels = d['labels']
        texts = [s['text'] for s in d['sentences']]
        log(f"[GNER-T5] {name}: {len(texts)} phrases (batch {batch_size}) ...")
        preds = _predict_corpus_batched(model, tokenizer, texts, labels, device,
                                        batch_size, max_new_tokens, max_input_length)
        for s, spans in zip(d['sentences'], preds):
            s['systems']['GNER-T5'] = _spans_with_surface(s['text'], spans)
        log(f"[GNER-T5] {name}: termine")
    del model


def run_qwen(dumps, max_eval, max_new_tokens):
    """Qwen2.5-1.5B int4, limite a --max-eval phrases par dataset (comme le
    benchmark). Les phrases au-dela ne recoivent aucune cle."""
    from scripts.baselines.run_llm_int4 import ADAPTERS, load_int4_model
    adapter = ADAPTERS['qwen1_5b']
    log(f"Chargement {adapter.checkpoint} (int4) ...")
    model, tok = load_int4_model(adapter.checkpoint)
    for name, d in dumps.items():
        labels = d['labels']
        n = min(max_eval, len(d['sentences']))
        log(f"[Qwen-1.5B] {name}: {n} phrases (sous-echantillon) ...")
        for i in range(n):
            s = d['sentences'][i]
            try:
                spans = adapter.predict(model, tok, s['text'], labels, max_new_tokens)
            except Exception as exc:                       # generation cassee
                log(f"    !! phrase {i} ignoree ({exc})")
                spans = []
            s['systems']['Qwen-1.5B'] = _spans_with_surface(s['text'], spans)
            if (i + 1) % 50 == 0:
                log(f"    ... {i + 1}/{n}")
    del model


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', nargs='+',
                    default=['e_ner', 'indian_legal', 'lener_br', 'german_ler_coarse'])
    ap.add_argument('--legal', action='store_true')
    ap.add_argument('--systems', nargs='+', default=['gliner_small', 'qwen1_5b', 'gner'],
                    choices=list(SYS_NAMES))
    ap.add_argument('--dump-dir', default='outputs/results/legal/error_analysis')
    ap.add_argument('--gliner-small', default='urchade/gliner_small-v2.1')
    ap.add_argument('--gner-checkpoint', default='dyyyyyyyy/GNER-T5-base')
    ap.add_argument('--threshold', type=float, default=0.3)
    ap.add_argument('--gner-batch-size', type=int, default=8)
    ap.add_argument('--max-new-tokens', type=int, default=640)
    ap.add_argument('--max-input-length', type=int, default=1024)
    ap.add_argument('--qwen-max-eval', type=int, default=200)
    args = ap.parse_args()

    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    log(f"=== Dump des baselines sur {device} ===")
    log(f"Systemes : {[SYS_NAMES[s] for s in args.systems]}")
    dumps = load_dumps(args.dump_dir, args.datasets)

    # Ordre volontaire : du moins cher au plus cher, pour que les premiers
    # resultats soient disponibles meme si le run long est interrompu.
    if 'gliner_small' in args.systems:
        run_gliner(dumps, args.gliner_small, args.threshold, device)
        save_dumps(args.dump_dir, dumps)
        if device == 'cuda':
            torch.cuda.empty_cache()

    if 'qwen1_5b' in args.systems:
        run_qwen(dumps, args.qwen_max_eval, args.max_new_tokens)
        save_dumps(args.dump_dir, dumps)
        if device == 'cuda':
            torch.cuda.empty_cache()

    if 'gner' in args.systems:
        run_gner(dumps, args.gner_checkpoint, device, args.gner_batch_size,
                 args.max_new_tokens, args.max_input_length)
        save_dumps(args.dump_dir, dumps)
        if device == 'cuda':
            torch.cuda.empty_cache()

    log("Termine. Analyses : python -m scripts.analyze_errors")


if __name__ == '__main__':
    main()
