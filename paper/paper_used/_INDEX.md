# Index des PDFs — `paper_used/` (OPENER-Legal)

PDFs **spécifiques au domaine juridique** ajoutés pour le workshop NLLP. Chaque PDF a été **vérifié** (titre/auteurs/venue lus sur la page de titre) et possède une entrée correcte dans `paper/references.bib`.

> Les briques techniques partagées (GLiNER, Nomic, Matryoshka, AMI, LinearSVC, CodeCarbon, baselines, Green-AI…) **réutilisent les PDFs déjà vérifiés du papier principal** (`../LyRIDS_Opener/paper/paper_used/`, correspondance 1:1 sur 60 réfs). Seuls les PDFs juridiques nouveaux sont rangés ici.

---

## `00_legal_nlp_survey/` — survey de cadrage (1)

| Fichier | Clé bib | Note |
|---|---|---|
| Natural Language Processing for the Legal Domain (Survey) | `ariai2025legalnlp` | Survey tâches/datasets/modèles/défis du NLP juridique. ACM Computing Surveys 2025 (to appear) ; PDF = arXiv:2410.21306v3. Cadre la RW-C. |

## `01_legal_ner_systems/` — systèmes de NER juridique (supervisés, par schéma/langue) (3)

| Fichier | Clé bib | Note |
|---|---|---|
| LegNER (domain-adapted transformer) | `karamitsos2025legner` | BERT domain-adapté + span supervision, NER légal anglais + anonymisation, 6 types (1 542 affaires). Frontiers in AI 8:1638971, 2025. **Approche « retrain par schéma » que l'on évite.** |
| NER for Serbian Legal Documents | `kalusev2025serbian` | BERT fine-tuné + nouveau dataset serbe (NER4Legal_SRB), 8 types, low-resource. ICIST 2025 ; PDF = arXiv:2502.10582. |
| FlairNLP at SemEval-2023 Task 6b | `ramesh2023flairnlp` | Système LegalEval (SemEval-2023 Task 6), Bi-LSTM + Flair, jugements indiens. arXiv:2306.02182 (pages proceedings non confirmées → cité en arXiv). |

## `02_domain_llm_ner/` — NER de domaine par LLM (1)

| Fichier | Clé bib | Note |
|---|---|---|
| Financial Named Entity Recognition: How Far Can LLM Go? | `lu2025financial` | Évalue GPT-4o / LLaMA-3.1 / Gemini-1.5 en NER financier, 5 types d'échecs. FinNLP-FNP-LLMFinLegal @ COLING 2025, pp. 164-168. Montre que même les LLM restent **schema-bound** + coûteux (contraste frugalité). |

---

## 📥 État

- **5 PDFs juridiques = 5 références** (`ariai2025legalnlp`, `karamitsos2025legner`, `kalusev2025serbian`, `ramesh2023flairnlp`, `lu2025financial`), tous cités dans la Related Work (§ Legal NLP and legal NER).
- **Datasets légaux évalués** (E-NER, LeNER-Br, German-LER) : réfs `au2022ener` / `luzdearaujo2018lenerbr` / `leitner2020germanler` (venues vérifiées via ACL Anthology/Springer). PDFs des papiers de datasets **à déposer** si besoin de relecture (E-NER déjà vérifié via ACL Anthology).
- **À fournir éventuellement** : PDFs de Legal-BERT (`chalkidis2020legalbert`) et LexGLUE (`chalkidis2022lexglue`) — venues déjà vérifiées via ACL Anthology, PDF non indispensable.
