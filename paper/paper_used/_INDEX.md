# Index des PDFs — `paper_used/` (OPENER-Legal, workshop NLLP)

Dossier **autonome** : il contient les PDFs de **toutes** les références citées par le workshop paper, y compris les briques techniques partagées avec le papier journal (recopiées depuis `../../LyRIDS_Opener/paper/paper_used/`, où elles ont été vérifiées titre/auteurs/venue). Plus besoin d'ouvrir le repo du journal pour relire une référence.

**État : 36 PDFs présents sur 38 références citées. 2 restent à fournir** (liste en bas, contexte optionnel uniquement).

Correspondance : 1 ligne = 1 clé `references.bib`. Vérification d'une entrée bib = ouvrir le PDF de la même ligne et lire la page de titre.

---

## `00_legal_nlp_survey/` — survey de cadrage (1)

| Fichier | Clé bib | Note |
|---|---|---|
| Natural Language Processing for the Legal Domain (Survey) | `ariai2025legalnlp` | Survey tâches/datasets/modèles/défis du NLP juridique. ACM Computing Surveys 2025 (to appear) ; PDF = arXiv:2410.21306v3. Cadre la Related Work légale. |

## `01_legal_ner_systems/` — systèmes de NER juridique (supervisés, par schéma/langue) (3)

| Fichier | Clé bib | Note |
|---|---|---|
| LegNER (domain-adapted transformer) | `karamitsos2025legner` | BERT domain-adapté + span supervision, NER légal anglais + anonymisation. Frontiers in AI 8:1638971, 2025. **Approche « retrain par schéma » que l'on évite.** |
| NER for Serbian Legal Documents | `kalusev2025serbian` | BERT fine-tuné + nouveau dataset serbe (NER4Legal_SRB), low-resource. ICIST 2025 ; PDF = arXiv:2502.10582. |
| FlairNLP at SemEval-2023 Task 6b | `ramesh2023flairnlp` | Système LegalEval, Bi-LSTM + Flair, jugements indiens. arXiv:2306.02182. |

## `02_domain_llm_ner/` — NER de domaine par LLM (1)

| Fichier | Clé bib | Note |
|---|---|---|
| Financial NER: How Far Can LLM Go? | `lu2025financial` | GPT-4o / LLaMA-3.1 / Gemini-1.5 en NER financier. FinNLP-FNP-LLMFinLegal @ COLING 2025. Montre que même les LLM restent **schema-bound** et coûteux. |

## `03_legal_datasets/` — les 4 corpus évalués (4 / 4) ✅

Les datasets du benchmark (Table 1 + section Datasets). 4 PDFs vérifiés page de titre le 2026-07-24.

| Fichier | Clé bib | Vérifié |
|---|---|---|
| E-NER — An Annotated NER Corpus of Legal Text | `au2022ener` | Au, Lampos, Cox. NLLP @ EMNLP 2022, pp. 246-255. ✅ page de titre OK. Confirme l'argument RW (« le NER généraliste chute de 29.4 à 60.4 % de F1 sur le légal »). |
| Named Entity Recognition in Indian court judgments | `kalamkar2022indianlegal` | Kalamkar et al. NLLP @ EMNLP 2022, pp. 184-193, 14 types légaux. ✅ page de titre OK. |
| LeNER-Br: a Dataset for NER in Brazilian Legal Text | `luzdearaujo2018lenerbr` | Luz de Araujo et al. PROPOR 2018 (pt). ✅ page de titre OK. |
| A Dataset of German Legal Documents for NER | `leitner2020germanler` | Leitner, Rehm, Moreno-Schneider (DFKI). LREC 2020, pp. 4478-4485 (PDF = arXiv:2003.13016). ✅ page de titre OK. Décisions de cours fédérales, schéma fin 19 classes / coarse 7. |

## `04_opener_pipeline/` — le pipeline réutilisé et ses briques (11)

| Fichier | Clé bib | Note |
|---|---|---|
| OPENER - KBS paper | `anon2026opener` | **Le papier journal généraliste** dont ce workshop est l'application légale. Cité anonymisé (double-blind). |
| GLiNER | `zaratiana2024gliner` | Détecteur de mentions frozen (S/M/L). |
| Nomic Embed | `nussbaum2024nomic` | Embedder Matryoshka réutilisé tel quel. |
| Matryoshka Representation Learning | `kusupati2022matryoshka` | Objectif de troncature de l'embedder. |
| FaceNet | `schroff2015facenet` | Origine du triplet margin loss du fine-tuning contrastif. |
| Sentence-BERT | `reimers2019sbert` | Librairie sentence-transformers. |
| LIBLINEAR | `fan2008liblinear` | Solveur du LinearSVC (OPENER-Sup). |
| Scikit-learn | `pedregosa2011scikit` | LinearSVC, AMI. |
| Information Theoretic Measures (AMI) | `vinh2010ami` | Métrique principale. |
| Quantifying the Carbon Emissions (CodeCarbon) | `lacoste2019codecarbon` | Mesure énergie/CO2. |
| CoNLL-2003 | `tjongkimsang2003conll` | Source d'entraînement (généraliste) de l'embedder, jamais du légal. |

## `05_baselines/` — systèmes comparés et leurs backbones (10)

| Fichier | Clé bib | Note |
|---|---|---|
| OWNER | `genest2025owner` | **Baseline la plus proche** (detect-then-type non supervisé). |
| GNER (Rethinking Negative Instances) | `ding2024gner` | Baseline générative. |
| Qwen2.5 Technical Report | `qwen2025` | Baseline LLM 4-bit. |
| DeBERTaV3 | `he2023debertav3` | Backbone de GLiNER. |
| T5 | `raffel2020t5` | Backbone de GNER. |
| QLoRA | `dettmers2023qlora` | Quantification NF4 4-bit. |
| UniversalNER | `zhou2024universalner` | LLM open-NER (discuté, non mesuré). |
| GoLLIE | `sainz2024gollie` | LLM open-NER (discuté, non mesuré). |
| ChatIE | `wei2024chatie` | Extraction par prompting (discuté). |
| Leveraging Type Descriptions | `aly2021zeroshot` | Typage zero-shot par sémantique des noms de types. |

## `06_energy_green_ai/` — axe frugalité (6)

| Fichier | Clé bib |
|---|---|
| Green AI | `schwartz2020greenai` |
| Energy and Policy Considerations for DL in NLP | `strubell2019energy` |
| The Carbon Footprint of ML Training | `patterson2022carbon` |
| Towards the Systematic Reporting of Energy/Carbon | `henderson2020systematic` |
| Sustainable AI | `wu2022sustainable` |
| Power Hungry Processing | `luccioni2024power` |

## `07_legal_models_benchmarks/` — contexte NLP juridique (0 / 2)

**Vide : optionnel.** Cités une fois chacun en Related Work comme contexte (« l'adaptation au domaine compte en NLU juridique »). Venues déjà vérifiées, PDF non indispensable.

| Attendu | Clé bib | Où le prendre |
|---|---|---|
| LEGAL-BERT | `chalkidis2020legalbert` | Findings of EMNLP 2020, pp. 2898-2904 (aussi arXiv:2010.02559) |
| LexGLUE | `chalkidis2022lexglue` | ACL 2022, pp. 4310-4330 (aussi arXiv:2110.00976) |

---

## 📥 À fournir (2, optionnel)

**Priorité basse — contexte, dossier `07_legal_models_benchmarks/`** : Legal-BERT (`chalkidis2020legalbert`, arXiv:2010.02559) et LexGLUE (`chalkidis2022lexglue`, arXiv:2110.00976). Cités une fois chacun en Related Work comme simple contexte, entrées bib déjà vérifiées via ACL Anthology, PDF non indispensable pour la relecture.

Les 36 autres références sont présentes et vérifiées. Les 4 datasets du benchmark sont au complet.
