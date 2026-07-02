#!/usr/bin/env bash
# =====================================================================
# Driver d'évaluation OPENER-Legal : les 4 points de fonctionnement du
# papier principal, appliqués aux datasets juridiques, embedder repris de HF.
#
# Lancer APRÈS le run multi-seed du papier principal (GPU 6 Go -> pas de
# contention). Activer le venv d'abord :
#   & c:\0-Code_py_temp\pytorch_cuda_env\Scripts\Activate.ps1
#   bash scripts/run_legal.sh
# (ou en PowerShell, lancer chaque commande python -m ... individuellement)
# =====================================================================
set -e

# Python du venv pytorch_cuda_env (bare `python` = Python système SANS torch/yaml).
PY="${PY:-/c/0-Code_py_temp/pytorch_cuda_env/Scripts/python.exe}"

# Embedder repris depuis HuggingFace (intent du papier : réutilisation, pas de
# ré-entraînement). Si le chargement HF échoue, basculer sur le poids local :
#   EMBEDDER=../LyRIDS_Opener/outputs/models/embedder_contrastive_hard_big bash scripts/run_legal.sh
EMBEDDER="${EMBEDDER:-Thibault-GAREL/opener-sup}"
MD="urchade/gliner_large-v2.1"
ANCHORS="configs/legal_anchors.yaml"
MAIN_DS="e_ner lener_br german_ler_coarse"   # table principale (schémas <= 7 types)
STRESS_DS="german_ler"                         # schéma fin 19 types (Analysis)
LOG="outputs/logs/legal_run_$(date +%Y%m%d_%H%M%S).log"
mkdir -p outputs/logs outputs/results/legal

echo "=== OPENER-Legal eval | embedder=$EMBEDDER | $(date) ===" | tee "$LOG"

# 1. OPENER-Sup — typing-on-gold (LinearSVC class_weight=balanced)
echo ">>> [1/4] OPENER-Sup gold" | tee -a "$LOG"
"$PY" -m scripts.run_balanced_classifiers --legal --embedder "$EMBEDDER" \
  --datasets $MAIN_DS $STRESS_DS \
  --output-dir outputs/results/legal/sup_gold 2>&1 | tee -a "$LOG"

# 2. OPENER-Sup — end-to-end (GLiNER détecteur + LinearSVC balanced + offsets/sentinels)
echo ">>> [2/4] OPENER-Sup e2e" | tee -a "$LOG"
"$PY" -m scripts.run_opener_e2e --legal --embedder "$EMBEDDER" --md-checkpoint "$MD" \
  --datasets $MAIN_DS --threshold 0.3 \
  --output-dir outputs/results/legal/sup_e2e 2>&1 | tee -a "$LOG"

# 3. OPENER-ZS — typing-on-gold (prototypes label-name, anchors EN via dict)
echo ">>> [3/4] OPENER-ZS gold" | tee -a "$LOG"
"$PY" -m scripts.run_opener_zs --legal --embedder "$EMBEDDER" \
  --datasets $MAIN_DS $STRESS_DS \
  --anchor-mode dict --anchor-dict "$ANCHORS" \
  --output-dir outputs/results/legal/zs_gold 2>&1 | tee -a "$LOG"

# 4. OPENER-ZS — end-to-end : transductif + fusion détecteur (β balayé, β=0.05 retenu)
echo ">>> [4/4] OPENER-ZS e2e + fusion" | tee -a "$LOG"
"$PY" -m scripts.run_opener_zs_e2e_fusion --legal --embedder "$EMBEDDER" --md-checkpoint "$MD" \
  --datasets $MAIN_DS \
  --anchor-mode dict --anchor-dict "$ANCHORS" --proto-mode ensemble --refine-iters 3 \
  --output-dir outputs/results/legal/zs_e2e_fusion 2>&1 | tee -a "$LOG"

echo "=== DONE $(date). Résultats : outputs/results/legal/ ===" | tee -a "$LOG"
