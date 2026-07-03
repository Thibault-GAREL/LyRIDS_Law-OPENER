#!/usr/bin/env bash
# =====================================================================
# Baselines du papier journal, évaluées sur les datasets JURIDIQUES
# (mêmes modèles que le benchmark général) : GLiNER S/M/L, GNER-T5-base,
# Qwen2.5-1.5B int4. Protocole end-to-end (détection + typing) + 3 axes.
# NB: gliner_small/medium + Qwen ont été supprimés du cache -> re-téléchargés
#     automatiquement au 1er run (~5 Go sur C:).
# Lancer venv actif :  bash scripts/run_legal_baselines.sh
# =====================================================================
set -u
PY="${PY:-/c/0-Code_py_temp/pytorch_cuda_env/Scripts/python.exe}"
DS="${DS:-e_ner lener_br german_ler_coarse}"
LOG="outputs/logs/legal_baselines_$(date +%Y%m%d_%H%M%S).log"
mkdir -p outputs/logs outputs/results/baselines
ts(){ date +'%H:%M:%S'; }

echo "[$(ts)] === Legal baselines (GLiNER S/M/L + GNER + Qwen-int4) on: $DS ===" | tee "$LOG"

# --- GLiNER small / medium / large (zero-shot, label-aware) ---
for ck in gliner_small-v2.1 gliner_medium-v2.1 gliner_large-v2.1; do
  tag="${ck%%-*}"
  echo "[$(ts)] --- GLiNER: $ck ---" | tee -a "$LOG"
  "$PY" -m scripts.baselines.run_gliner --legal --checkpoint "urchade/$ck" \
      --datasets $DS --output-dir "outputs/results/baselines/$tag" 2>&1 | tee -a "$LOG"
done

# --- GNER-T5-base (generative NER, zero-shot) ---
echo "[$(ts)] --- GNER-T5-base ---" | tee -a "$LOG"
"$PY" -m scripts.baselines.run_gner --legal --checkpoint dyyyyyyyy/GNER-T5-base \
    --datasets $DS --output-dir outputs/results/baselines/gner --resume 2>&1 | tee -a "$LOG"

# --- Qwen2.5-1.5B int4 (LLM généraliste quantifié) ---
# Génératif -> plus lent : échantillon test réduit (comme le journal), documenté.
echo "[$(ts)] --- Qwen2.5-1.5B int4 (max-eval 200) ---" | tee -a "$LOG"
"$PY" -m scripts.baselines.run_llm_int4 --legal --model qwen1_5b \
    --datasets $DS --max-eval 200 --resume 2>&1 | tee -a "$LOG"

echo "[$(ts)] === DONE. Résultats: outputs/results/baselines/ ===" | tee -a "$LOG"
