#!/usr/bin/env bash
# =====================================================================
# Baseline OWNER sur les datasets JURIDIQUES (protocole transfert du journal).
#
# ⚠️ OWNER (codebase + checkpoint) est un SOUS-MODULE git du repo PRINCIPAL
#    (external/OWNER, ~plusieurs Go partages avec le papier journal). On ne le
#    duplique PAS ici : ce script le REFERENCE via $OWNER_REPO. Seuls les
#    artefacts specifiques au legal (resultats, log) vivent dans ce projet.
#
# Flux : export legal -> format OWNER  ->  gen 3 configs TOML (load_finetuned
#        depuis checkpoints/transfer_conll2003/100)  ->  inference OWNER (env
#        conda separe)  ->  collect AMI.  Aucun re-entrainement d'OWNER.
#
# Lancer venv actif :  bash scripts/run_legal_owner.sh
# =====================================================================
set -u
PY="${PY:-/c/0-Code_py_temp/pytorch_cuda_env/Scripts/python.exe}"   # venv (export + collect)
OWNERPY="D:/conda_envs/owner/python.exe"                            # env conda OWNER (inference)
OWNER_REPO="../LyRIDS_Opener/external/OWNER"                        # sous-module partage (repo principal)
OWNER_ABS="D:/Loisir/Code_python/LyRIDS_Opener/external/OWNER"
DS="e_ner lener_br german_ler_coarse"
CFGDIR="$OWNER_REPO/configs/lyrids_ner_transfer"
LOG="outputs/logs/legal_owner_$(date +%Y%m%d_%H%M%S).log"
mkdir -p outputs/logs outputs/results/baselines/owner
ts(){ date +'%H:%M:%S'; }

echo "[$(ts)] === OWNER legal (transfer, load_finetuned conll2003) ===" | tee "$LOG"

# 1) Export des datasets legaux au format OWNER (dans le sous-module).
"$PY" -m scripts.baselines.owner_export --legal --datasets $DS \
    --out "$OWNER_REPO/data/lyrids" --max-test 1000 2>&1 | tee -a "$LOG"

# 2) Genere une config TOML par dataset (copie le template fabner, change le test).
"$PY" - "$OWNER_ABS" $DS <<'PYGEN' 2>&1 | tee -a "$LOG"
import sys, pathlib
owner_abs = sys.argv[1]; datasets = sys.argv[2:]
cfgdir = pathlib.Path(owner_abs)/'configs'/'lyrids_ner_transfer'
tmpl = (cfgdir/'fabner.toml').read_text(encoding='utf-8')
base = f'{owner_abs}/data/lyrids'
for ds in datasets:
    s = tmpl.replace(f'{base}/fabner/test.json', f'{base}/{ds}/test.json')
    s = s.replace('test_dataset_name = "fabner"', f'test_dataset_name = "{ds}"')
    (cfgdir/f'{ds}.toml').write_text(s, encoding='utf-8'); print('config ->', ds)
PYGEN

# 3) Inference OWNER (env conda), 1 config par dataset.
export MLFLOW_TRACKING_URI="file:///$OWNER_ABS/mlruns"
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"
( cd "$OWNER_REPO" && for ds in $DS; do
    echo "[$(date +%H:%M:%S)] --- OWNER: $ds ---"
    "$OWNERPY" -u -m owner.main --config-file="configs/lyrids_ner_transfer/$ds.toml"
    echo "[$(date +%H:%M:%S)] $ds exit=$?"
  done ) 2>&1 | tee -a "$LOG"

# 4) Collect AMI depuis le file store mlflow -> outputs/results/baselines/owner/.
"$PY" -m scripts.baselines.owner_collect --mlruns "$OWNER_REPO/mlruns" \
    --output-dir outputs/results/baselines/owner 2>&1 | tee -a "$LOG"

echo "[$(ts)] === OWNER legal DONE ===" | tee -a "$LOG"
