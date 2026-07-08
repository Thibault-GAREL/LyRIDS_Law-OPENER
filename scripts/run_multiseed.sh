#!/usr/bin/env bash
# =====================================================================
# Étude multi-seed OPENER-Legal (reviewer : "single random seed").
#
# RIEN n'est ré-entraîné ici : l'embedder est repris FIGÉ des ré-entraînements
# multi-seed du repo principal (../LyRIDS_Opener/outputs/models/seed{S}_hard,
# même recette contrastif + hard-mining que l'embedder released seed 42).
# Pour chaque seed, on réévalue les 4 têtes du paper sur le benchmark légal :
#   sup_gold  run_balanced_classifiers          (LinearSVC balanced)
#   zs_gold   run_opener_zs (dict)              (nearest label-name centroid)
#   sup_e2e   run_opener_e2e GLiNER-M           (+ --fit-on-detected, t=0.3)
#   zs_e2e    run_opener_zs_e2e_fusion GLiNER-L (ensemble + refine 3, beta 0.05)
# Le seed 42 (released) = les runs canoniques déjà dans le paper, PAS re-run.
#
# Règles frugales (héritées des incidents du repo principal) :
#  - un process python PAR DATASET (la RAM est rendue à l'OS entre datasets) ;
#  - resume fin : skip si le SUMMARY du dataset existe déjà ;
#  - HF offline (une micro-coupure réseau nocturne a déjà tué un run) ;
#  - threads CPU limités + priorité BelowNormal (PC utilisable pendant le run).
#
# Usage :  bash scripts/run_multiseed.sh            # seeds 7 et 123
#          bash scripts/run_multiseed.sh 7          # un seul seed
# Résultats -> outputs/results/seed_runs/seed{S}/{tête}/{dataset}/
# Agrégation ensuite : python -m scripts.aggregate_multiseed [--latex]
# =====================================================================
set -u
PY="${PY:-/c/0-Code_py_temp/pytorch_cuda_env/Scripts/python.exe}"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
# Modèles (GLiNER M/L) et datasets HF déjà dans le cache local des runs seed 42.
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1

MAIN_DS="e_ner indian_legal lener_br german_ler_coarse"
STRESS_DS="german_ler"                       # 19 types, têtes gold uniquement
ANCHORS="configs/legal_anchors.yaml"
MD_SUP="urchade/gliner_medium-v2.1"
MD_ZS="urchade/gliner_large-v2.1"
EMB_ROOT="../LyRIDS_Opener/outputs/models"
SEEDS="${1:-7 123}"

LOGDIR="/c/0-Code_py_temp/0-log_progress/$(date +%Y-%m-%d)-OPENER_Legal-01"
mkdir -p "$LOGDIR"
LOG="$LOGDIR/multiseed_$(date +%Y%m%d_%H%M%S).log"
ts() { date +'%Y-%m-%d %H:%M:%S'; }

# Priorité BelowNormal sur les python du venv, réappliquée toutes les 60 s
# (les process sont recréés à chaque dataset).
( while true; do
    powershell.exe -NoProfile -Command \
      "Get-Process python -ErrorAction SilentlyContinue | ForEach-Object { try { \$_.PriorityClass = 'BelowNormal' } catch {} }" \
      >/dev/null 2>&1
    sleep 60
  done ) &
WATCHER=$!
trap 'kill $WATCHER 2>/dev/null' EXIT

# run_eval <module> <outdir> <dataset> [args...] : skip si SUMMARY présent.
run_eval() {
  local module=$1 outdir=$2 dataset=$3
  shift 3
  if ls "$outdir/$dataset"/SUMMARY*.json >/dev/null 2>&1; then
    echo "[$(ts)]     $dataset : SUMMARY présent -> skip" | tee -a "$LOG"
    return 0
  fi
  mkdir -p "$outdir/$dataset"
  echo "[$(ts)]     $dataset : run..." | tee -a "$LOG"
  "$PY" -m "$module" --legal --datasets "$dataset" --output-dir "$outdir/$dataset" "$@" \
    >>"$LOG" 2>&1 || echo "[$(ts)] !! FAIL $module / $dataset" | tee -a "$LOG"
}

echo "[$(ts)] multi-seed legal START | seeds: $SEEDS | log: $LOG" | tee -a "$LOG"

for S in $SEEDS; do
  EMB="$EMB_ROOT/seed${S}_hard"
  RES="outputs/results/seed_runs/seed${S}"
  if [ ! -f "$EMB/model.safetensors" ]; then
    echo "[$(ts)] !! seed $S : embedder absent ($EMB), skip" | tee -a "$LOG"
    continue
  fi
  echo "[$(ts)] ================= SEED $S (embedder: $EMB) =================" | tee -a "$LOG"

  echo "[$(ts)] [1/4] sup_gold" | tee -a "$LOG"
  for ds in $MAIN_DS $STRESS_DS; do
    run_eval scripts.run_balanced_classifiers "$RES/sup_gold" "$ds" --embedder "$EMB"
  done

  echo "[$(ts)] [2/4] zs_gold" | tee -a "$LOG"
  for ds in $MAIN_DS $STRESS_DS; do
    run_eval scripts.run_opener_zs "$RES/zs_gold" "$ds" --embedder "$EMB" \
      --anchor-mode dict --anchor-dict "$ANCHORS"
  done

  echo "[$(ts)] [3/4] sup_e2e (GLiNER-M, fit-on-detected)" | tee -a "$LOG"
  for ds in $MAIN_DS; do
    run_eval scripts.run_opener_e2e "$RES/sup_e2e" "$ds" --embedder "$EMB" \
      --md-checkpoint "$MD_SUP" --threshold 0.3 --fit-on-detected
  done

  echo "[$(ts)] [4/4] zs_e2e (GLiNER-L, fusion)" | tee -a "$LOG"
  for ds in $MAIN_DS; do
    run_eval scripts.run_opener_zs_e2e_fusion "$RES/zs_e2e" "$ds" --embedder "$EMB" \
      --md-checkpoint "$MD_ZS" --anchor-mode dict --anchor-dict "$ANCHORS" \
      --proto-mode ensemble --refine-iters 3
  done

  echo "[$(ts)] ================= SEED $S : DONE =================" | tee -a "$LOG"
done

echo "[$(ts)] multi-seed legal DONE. Agrégation : python -m scripts.aggregate_multiseed" | tee -a "$LOG"
