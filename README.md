# ⚖️ Law-OPENER: a Frugal Open-World NER Pipeline for the Legal Domain

![Python](https://img.shields.io/badge/python-3.10-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1%2Bcu121-red.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.7-orange.svg)
![CUDA](https://img.shields.io/badge/CUDA-12.1-76B900.svg)
![Sentence-Transformers](https://img.shields.io/badge/sentence--transformers-5.5-purple.svg)
![GLiNER](https://img.shields.io/badge/GLiNER-M%2FL--v2.1-lightgrey.svg)

[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hub-opener--sup-yellow.svg)](https://huggingface.co/Thibault-GAREL/opener-sup)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hub-opener--zs-yellow.svg)](https://huggingface.co/Thibault-GAREL/opener-zs)
[![Paper](https://img.shields.io/badge/ACL%20Anthology-NLLP%202026%20%40%20EMNLP-ED1C24.svg)](https://aclanthology.org/2026.nllp-1.40/)
[![Talk](https://img.shields.io/badge/YouTube-NLLP%202026%20live%20talk-FF0000.svg?logo=youtube&logoColor=white)](https://youtu.be/oUS7c9HGJ_k?t=19013)

![License](https://img.shields.io/badge/license-MIT-green.svg)
![Contributions](https://img.shields.io/badge/contributions-welcome-orange.svg)

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/readme/legal-fig-cost-dark.png">
    <img src="assets/readme/legal-fig-cost-light.png" alt="End-to-end AMI against energy on the four legal corpora" width="85%">
  </picture>
</p>

---

## 📝 Project Description

**Law-OPENER** takes [OPENER](https://github.com/Thibault-GAREL/LyRIDS_OPENER), my frugal open-world NER pipeline built on general-domain text, and applies it **unchanged** to legal NER in English, Portuguese and German, with no component retrained on legal data. The study is a **workshop paper published at [NLLP 2026](https://aclanthology.org/2026.nllp-1.40/)** (EMNLP), measured on quality (AMI), latency and energy at once.

---

## 📄 Full Paper

<p align="center">
  📄 <a href="https://aclanthology.org/2026.nllp-1.40/"><b>Read it on the ACL Anthology</b></a> &nbsp;·&nbsp; 🎥 <a href="https://youtu.be/oUS7c9HGJ_k?t=19013"><b>Watch the live talk at NLLP 2026</b></a>
</p>

<p align="center">
  <img src="assets/paper_pages/page-01.png" alt="Law-OPENER paper, page 01" width="800">
</p>

<details>
<summary>📄 <b>Show the remaining 14 pages (NLLP 2026 proceedings)</b></summary>

<p align="center">
  <img src="assets/paper_pages/page-02.png" alt="Law-OPENER paper, page 02" width="800">
</p>

<p align="center">
  <img src="assets/paper_pages/page-03.png" alt="Law-OPENER paper, page 03" width="800">
</p>

<p align="center">
  <img src="assets/paper_pages/page-04.png" alt="Law-OPENER paper, page 04" width="800">
</p>

<p align="center">
  <img src="assets/paper_pages/page-05.png" alt="Law-OPENER paper, page 05" width="800">
</p>

<p align="center">
  <img src="assets/paper_pages/page-06.png" alt="Law-OPENER paper, page 06" width="800">
</p>

<p align="center">
  <img src="assets/paper_pages/page-07.png" alt="Law-OPENER paper, page 07" width="800">
</p>

<p align="center">
  <img src="assets/paper_pages/page-08.png" alt="Law-OPENER paper, page 08" width="800">
</p>

<p align="center">
  <img src="assets/paper_pages/page-09.png" alt="Law-OPENER paper, page 09" width="800">
</p>

<p align="center">
  <img src="assets/paper_pages/page-10.png" alt="Law-OPENER paper, page 10" width="800">
</p>

<p align="center">
  <img src="assets/paper_pages/page-11.png" alt="Law-OPENER paper, page 11" width="800">
</p>

<p align="center">
  <img src="assets/paper_pages/page-12.png" alt="Law-OPENER paper, page 12" width="800">
</p>

<p align="center">
  <img src="assets/paper_pages/page-13.png" alt="Law-OPENER paper, page 13" width="800">
</p>

<p align="center">
  <img src="assets/paper_pages/page-14.png" alt="Law-OPENER paper, page 14" width="800">
</p>

<p align="center">
  <img src="assets/paper_pages/page-15.png" alt="Law-OPENER paper, page 15" width="800">
</p>

</details>

Full PDF (text-selectable): [`OPENER-Legal_NLLP2026_ACL-Anthology.pdf`](paper/OPENER-Legal_NLLP2026_ACL-Anthology.pdf)

---

## ⚙️ Features

  🔁 **Zero retraining**: the detector and the embedder are reused as released, only the typing head ever sees legal data.

  🎯 **Two operating points** on one embedder, **OPENER-ZS** (no target labels at all) and **OPENER-Sup** (a balanced linear probe).

  🌍 **Four legal corpora, three languages**: E-NER and Indian-Legal (en), LeNER-Br (pt), German-LER (de).

  🔋 **Three axes at once** against GLiNER S/M/L, GNER, OWNER and a 4-bit Qwen: AMI, latency and energy (CodeCarbon).

  📏 **Uncertainty reported**: 3 embedder seeds plus a paired bootstrap (K = 1000) on every test set.

---

## Example Outputs

### 🎯 Typing on gold mentions

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/readme/legal-fig-gold-dark.png">
    <img src="assets/readme/legal-fig-gold-light.png" alt="Typing-on-gold AMI per legal corpus for OWNER, OPENER-ZS and OPENER-Sup" width="85%">
  </picture>
</p>

The supervised probe holds across languages (**61.4** mean AMI), while the label-free head drops from English to German.

<details>
<summary>📊 <b>Gold typing table (AMI and macro-F1, ×100)</b></summary>

| System | E-NER (en) | Indian (en) | LeNER-Br (pt) | German (de) | Mean |
|---|--:|--:|--:|--:|--:|
| *AMI* | | | | | |
| OWNER | 54.5 | 59.0 | 39.7 | 25.7 | 44.7 |
| OPENER-ZS | 47.7 ± 3.5 | 49.1 ± 3.0 | 37.4 ± 6.0 | 23.8 ± 4.3 | 39.5 ± 3.0 |
| **OPENER-Sup** | **70.9 ± 1.2** | **60.8 ± 1.4** | **53.7 ± 1.2** | **60.3 ± 1.0** | **61.4 ± 0.4** |
| *Macro-F1* | | | | | |
| OPENER-ZS | 59.2 ± 7.0 | 37.6 ± 5.3 | 57.7 ± 6.0 | 29.7 ± 5.4 | 46.1 ± 5.3 |
| **OPENER-Sup** | **89.1 ± 0.2** | **58.3 ± 2.5** | **74.7 ± 1.5** | **76.3 ± 1.3** | **74.6 ± 0.3** |

OPENER cells are mean ± std over 3 embedder seeds. OWNER names its own clusters, so no F1 is defined for it.

</details>

### 📈 End-to-end, per corpus (AMI ×100)

| Model | E-NER (en) | Indian (en) | LeNER-Br (pt) | German (de) | Avg |
|---|--:|--:|--:|--:|--:|
| GLiNER-S | 33.4 | 33.9 | 33.7 | 31.1 | 33.0 |
| GLiNER-M | 33.9 | 34.8 | 34.8 | 36.1 | 34.9 |
| GLiNER-L | 34.7 | **35.2** | **36.3** | 34.0 | 35.0 |
| GNER-T5 | 28.7 | 28.6 | 33.2 | **39.2** | 32.4 |
| Qwen2.5-1.5B (int4) | 24.9 | 28.1 | 24.9 | 14.5 | 23.1 |
| OWNER | **38.4** | 33.3 | 34.8 | 28.4 | 33.7 |
| OPENER-ZS | 33.4 ± 0.2 | 33.8 ± 0.5 | 33.5 ± 1.2 | 32.8 ± 0.4 | 33.4 ± 0.4 |
| **OPENER-Sup** | 35.7 ± 0.3 | **35.2 ± 0.3** | 34.1 ± 0.7 | 38.5 ± 0.3 | **35.9 ± 0.4** |

Every system sits in a narrow 33 to 36 band, because the English detectors find only part of the legal mentions. **Bold** is the best per column, and only OPENER-Sup uses target labels.

<details>
<summary>💸 <b>Quality against cost (params, AMI, F1, latency, energy)</b></summary>

| Model | Params | AMI ↑ | F1 ↑ | p50 (ms) ↓ | Energy (Wh / 1000 sentences) ↓ |
|---|--:|--:|--:|--:|--:|
| GLiNER-S | 50M | 33.0 | 22.2 | **22** | **0.5** |
| GLiNER-M | 200M | 34.9 | 22.8 | 37 | 0.7 |
| GLiNER-L | 330M | 35.0 | 25.0 | 81 | 1.6 |
| GNER-T5 | 220M | 32.4 | 18.4 | 8289 | 96.0 |
| Qwen2.5-1.5B (int4) | 1.5B | 23.1 | 10.4 | 5426 | 100.5 |
| OWNER | ≈294M | 33.7 | n/a | 475 | 14.6 |
| OPENER-ZS | 467M | 33.4 | 23.1 | 133 | 2.7 |
| **OPENER-Sup** | 337M | **35.9** | **25.3** | 84 | 1.7 |

Four-corpus means. Only AMI is a three-seed mean, F1, latency and energy come from the released seed. Params count the whole pipeline (detector plus the 137M embedder). Qwen runs on a 200-sentence subsample per corpus.

</details>

<details>
<summary>🔬 <b>Where end-to-end quality is lost (detection vs typing)</b></summary>

| System | Detected, exact (% gold) | Detected, touched (% gold) | Typing acc. | F1 | AMI | Oracle AMI |
|---|--:|--:|--:|--:|--:|--:|
| GLiNER-L | 41.1 | 68.6 | 70.1 | 25.0 | 35.0 | 72.0 |
| ↳ OPENER-ZS | 41.1 | 68.6 | 64.5 | 23.1 | 33.7 | 72.0 |
| GLiNER-M | 35.7 | 63.4 | 64.6 | 22.8 | 34.9 | 69.8 |
| ↳ **OPENER-Sup** | 35.7 | 63.4 | **78.7** | **25.3** | **35.9** | 69.8 |

- **Same spans, different head**: each ↳ row runs on the detector above it, so only the typing head changes.
- **Typing is not the bottleneck**: OPENER-Sup types 78.7% of the detected mentions right, against 64.6% for GLiNER-M alone.
- **Detection is the ceiling**: perfect typing of these spans would still stop around 70 AMI (oracle column).
- Released seed only, so the AMI column can differ slightly from the three-seed means above.

</details>

<details>
<summary>📝 <b>Notes and observations</b></summary>

- **OPENER-ZS is seed sensitive**: its gold AMI moves by ±3 to 6 across embedder seeds, the probe by at most ±1.4.
- **The lead is significant**: a paired bootstrap gives OPENER-Sup +1.0 AMI over GLiNER-M (95% CI [+0.4, +1.7]) and over GLiNER-L ([+0.1, +1.9]).
- **German fails as a language, not as a schema**: GLiNER-L touches 94% of the Portuguese legislation spans, but only 42% of the German persons (1.2 words on average).
- **Fine schema stress test**: on the 19 fine German-LER types, OPENER-Sup still reaches 61.5 ± 0.7 gold AMI, OPENER-ZS falls to 22.7 ± 2.1.
- **Typical label-free error**: OPENER-ZS maps NORM onto REGULATION 213 times on German-LER (the probe does it 3 times), since both names are near synonyms in English.
- **Fusion weight β**: OPENER-ZS goes from 32.4 AMI at β = 0 up to the GLiNER-L labels (35.0), so it cannot beat the detector it fuses with.

</details>

---

## ⚙️ How it works

  🔍 **Detect**: a frozen GLiNER (M for OPENER-Sup, L for OPENER-ZS) proposes spans, prompted with the legal type names.

  🧠 **Embed**: each span is embedded inside its sentence by the OPENER Nomic v1.5 embedder, pulled from the Hugging Face Hub as is.

  🏷️ **Type with labels (OPENER-Sup)**: a balanced `LinearSVC` is fitted on the frozen embeddings of the target's training mentions.

  🧭 **Type without labels (OPENER-ZS)**: English label-name prototypes, refined on the unlabelled test mentions, then fused with the detector's own label.

  📏 **Score**: AMI over offset-aligned mentions, with sentinel classes so detection errors stay in the score.

<details>
<summary>🧩 <b>Span in context and type anchors</b></summary>

- The embedder reads the whole sentence with the mention marked in place, `classification: The Court applied [ENT] Section 10(b) [/ENT] of the Exchange Act.`, so the same name can get a different type in two clauses.
- OPENER-ZS needs one vector per type, and codes like `ORGANIZACAO` or `GS` mean little to an English encoder. Each type gets 1 to 5 English anchor words, fixed before any result was seen (`COURT` → court, tribunal, judiciary), listed in [`configs/legal_anchors.yaml`](configs/legal_anchors.yaml).

</details>

<details>
<summary>📏 <b>Evaluation protocol</b></summary>

- Predicted and gold mentions are aligned on exact character offsets. A missed gold mention and a spurious span each fall into a sentinel class.
- Quality is the **Adjusted Mutual Information** between the gold and predicted partitions. It needs no label matching, so OWNER (which names its own clusters) is scored the same way.
- Each test split is capped at its first 1000 entity-bearing sentences. Every hyperparameter is fixed a priori, none is tuned on a legal set.
- OPENER cells are mean ± std over 3 embedder seeds (7, 42, 123). Latency and energy come from the released seed 42, since a seed changes the weights and not the FLOPs.

</details>

---

## 🗺️ Architecture Diagram

<p align="center">
  <img src="assets/opener-architecture-v3.svg" alt="OPENER architecture applied to legal text" width="85%">
</p>

Same pipeline as OPENER, nothing retrained on legal data. Only the typing head changes between the two operating points.

<details>
<summary>📐 <b>Key hyperparameters</b></summary>

- **Detector**: `urchade/gliner_medium-v2.1` for OPENER-Sup, `urchade/gliner_large-v2.1` for OPENER-ZS, threshold 0.3, frozen.
- **Embedder**: `Thibault-GAREL/opener-sup` (same weights as `opener-zs`), full 768 dims, no Matryoshka truncation.
- **OPENER-Sup**: `LinearSVC(C=1, class_weight='balanced')`, refitted end-to-end on the detector's own training spans (`--fit-on-detected`).
- **OPENER-ZS**: anchor words embedded through 4 templates, 3 iterations of spherical k-means on the test mentions, fusion weight β = 0.05.
- Full configuration in [`configs/legal.yaml`](configs/legal.yaml).

</details>

---

## 📚 Legal Datasets

| Dataset | Lang | Domain | Types | Test sentences | Mentions | Source |
|---|---|---|--:|--:|--:|---|
| **E-NER** | en | U.S. SEC/EDGAR filings | 7 | 877 | 1,793 | [GitHub `terenceau1/E-NER-Dataset`](https://github.com/terenceau1/E-NER-Dataset) |
| **Indian-Legal** | en | Indian court judgments | 14 | 866 | 3,028 | 🤗 [`AjayMukundS/Indian_Legal_NER_Dataset`](https://huggingface.co/datasets/AjayMukundS/Indian_Legal_NER_Dataset) |
| **LeNER-Br** | pt | Brazilian court decisions | 6 | 636 | 1,536 | 🤗 [`peluz/lener_br`](https://huggingface.co/datasets/peluz/lener_br) |
| **German-LER** | de | German federal court decisions | 7 (19 fine) | 1,000 | 1,740 | 🤗 [`elenanereiss/german-ler`](https://huggingface.co/datasets/elenanereiss/german-ler) |

The two English sets isolate a pure **domain** shift, LeNER-Br and German-LER add a **cross-lingual** one. All four are downloaded on first use by [`src/data/legal_datasets.py`](src/data/legal_datasets.py).

<details>
<summary>🗑️ <b>Datasets tested, then dropped</b></summary>

- **MAPA** (en and fr) and **CUAD** were evaluated, then left out of the paper. MAPA annotates generic anonymisation entities and CUAD contract clauses, so neither is legal-specific NER.
- E-NER ships without an official split. It is split 80/20 with seed 42, and Indian-Legal is evaluated on its `validation` split.

</details>

---

## 📂 Repository structure

<details>
<summary>📂 <b>Show the file tree</b></summary>

```bash
LyRIDS_Law-OPENER/
├── assets/
│   ├── opener-architecture-v3.svg     # pipeline figure (README and paper)
│   ├── readme/                        # README figures, light and dark
│   └── paper_pages/                   # published pages as PNG
│
├── configs/
│   ├── legal.yaml                     # embedder, detectors, typing heads, datasets
│   └── legal_anchors.yaml             # English anchor words per legal type (OPENER-ZS)
│
├── src/
│   ├── data/
│   │   ├── legal_datasets.py          # loaders for the 4 legal corpora (auto-download)
│   │   ├── owner_datasets.py          # shared span format and registry
│   │   └── schema.py
│   ├── models/
│   │   ├── embedder.py                # Nomic Matryoshka wrapper (span in context)
│   │   └── mention_detector.py        # GLiNER wrapper
│   └── utils/                         # config, energy (CodeCarbon), timing (p50/p95/p99)
│
├── scripts/
│   ├── run_balanced_classifiers.py    # OPENER-Sup on gold mentions
│   ├── run_opener_e2e.py              # OPENER-Sup end-to-end
│   ├── run_opener_zs.py               # OPENER-ZS on gold mentions
│   ├── run_opener_zs_e2e_fusion.py    # OPENER-ZS end-to-end (transductive + fusion)
│   ├── run_bootstrap_ci.py            # paper numbers, gold (bootstrap)
│   ├── run_bootstrap_e2e.py           # paper numbers, end-to-end (paired bootstrap)
│   ├── aggregate_multiseed.py         # 3 seeds -> mean ± std
│   ├── verify_paper_numbers.py        # every OPENER cell of the paper vs the JSON results
│   ├── verify_hf_embedder.py          # HF weights vs local checkpoint, bit for bit
│   ├── analyze_errors.py              # F1, oracle ceiling, confusion analysis
│   ├── make_readme_assets.py          # README figures and paper pages
│   ├── run_legal.sh                   # the 4 operating points on the 4 corpora
│   ├── run_legal_baselines.sh         # GLiNER S/M/L, GNER, Qwen int4
│   ├── run_legal_owner.sh             # OWNER (submodule of the main OPENER repo)
│   ├── run_multiseed.sh               # seeds 7 and 123
│   └── baselines/                     # baseline runners
│
├── paper/
│   ├── main.tex                       # ACL template
│   ├── sections/                      # one .tex per section
│   ├── references.bib
│   ├── paper_used/                    # reference PDFs, sorted by topic
│   ├── OPENER-Legal_NLLP2026_camera-ready.pdf
│   └── OPENER-Legal_NLLP2026_ACL-Anthology.pdf   # published version (proceedings pages 469 to 483)
│
├── outputs/                           # logs, energy, results (results gitignored)
├── data/                              # raw and processed corpora (gitignored)
│
├── LICENSE
└── README.md
```

</details>

---

## 💻 Run it on Your PC
Clone the repository and install dependencies:
```bash
git clone https://github.com/Thibault-GAREL/LyRIDS_Law-OPENER.git
cd LyRIDS_Law-OPENER

python -m venv .venv # if you don't have a virtual environment
source .venv/bin/activate   # Linux / macOS
.venv\Scripts\activate      # Windows

pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install gliner sentence-transformers einops scikit-learn pyyaml datasets codecarbon

# OPENER-Sup on the E-NER gold mentions, embedder pulled from the Hugging Face Hub
python -m scripts.run_balanced_classifiers --legal --embedder Thibault-GAREL/opener-sup --datasets e_ner --output-dir outputs/results/legal/sup_gold
```

⚠️ A **CUDA-compatible GPU** is recommended (a 6 GB card is enough). Everything also runs on CPU, much slower.

<details>
<summary>▶️ <b>Run the 4 operating points on the 4 corpora</b></summary>

```bash
# Sup and ZS, on gold mentions and end-to-end, with a timestamped log in outputs/logs/
PY=python bash scripts/run_legal.sh   # on Windows, run it from Git Bash
```

</details>

<details>
<summary>📏 <b>Reproduce the paper numbers</b></summary>

```bash
# gold typing, mean ± std and 95% CI (bootstrap, K = 1000)
python -m scripts.run_bootstrap_ci --legal --embedder Thibault-GAREL/opener-sup --datasets e_ner indian_legal lener_br german_ler_coarse

# end-to-end, paired bootstrap between systems on the same sentences
python -m scripts.run_bootstrap_e2e --legal --embedder Thibault-GAREL/opener-sup

# check every OPENER cell of the paper against the aggregated results
python -m scripts.verify_paper_numbers
```

The seed 7 and seed 123 embedders come from the main OPENER repo and are not on the Hub, so `scripts/run_multiseed.sh` needs that repo cloned next to this one.

</details>

<details>
<summary>🧪 <b>Run the baselines</b></summary>

```bash
# GLiNER S/M/L, GNER-T5 and Qwen2.5-1.5B in 4-bit (needs bitsandbytes)
pip install bitsandbytes
PY=python bash scripts/run_legal_baselines.sh

# OWNER reuses the submodule and the conda env of the main OPENER repo, no retraining
bash scripts/run_legal_owner.sh
```

</details>

<details>
<summary>🖼️ <b>Regenerate the README figures</b></summary>

```bash
pip install matplotlib pymupdf
python -m scripts.make_readme_assets
```

</details>

---

## 📖 Inspiration / Sources
This project is based on:
- 🔗 [LyRIDS OPENER](https://github.com/Thibault-GAREL/LyRIDS_OPENER), the general-domain pipeline and embedder reused here unchanged.
- 📄 [GLiNER](https://github.com/urchade/GLiNER), the frozen zero-shot detector.
- 📄 [Nomic Embed Text v1.5](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5), the Matryoshka embedder.
- 📄 [OWNER](https://doi.org/10.1109/ACCESS.2025.3552122), the open-world baseline (my own re-implementation: [LyRIDS OWNER recreating](https://github.com/Thibault-GAREL/LyRIDS_OWNER_recreating)).
- 📄 The four legal corpora: [E-NER](https://doi.org/10.18653/v1/2022.nllp-1.22), [Indian-Legal](https://doi.org/10.18653/v1/2022.nllp-1.15), [LeNER-Br](https://doi.org/10.1007/978-3-319-99722-3_32) and [German-LER](https://aclanthology.org/2020.lrec-1.551/).

Paper: 📄 [ACL Anthology](https://aclanthology.org/2026.nllp-1.40/) (local copy: [`OPENER-Legal_NLLP2026_ACL-Anthology.pdf`](paper/OPENER-Legal_NLLP2026_ACL-Anthology.pdf)), co-authored with Faiza Belbachir and Assia Soukane (LyRIDS, ECE Paris).

Talk: 🎥 [NLLP 2026 live on YouTube](https://youtu.be/oUS7c9HGJ_k?t=19013), starting at the Law-OPENER presentation.

Embedders on the Hugging Face Hub: 🤗 [`opener-sup`](https://huggingface.co/Thibault-GAREL/opener-sup) and 🤗 [`opener-zs`](https://huggingface.co/Thibault-GAREL/opener-zs).

Code created by me 😎, Thibault GAREL - [Github](https://github.com/Thibault-GAREL)
