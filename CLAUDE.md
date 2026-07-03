# CLAUDE.md — OPENER-Legal (workshop paper)

Projet **frère** de `LyRIDS_Opener`. Objectif : **appliquer OPENER au NER juridique** (legal domain) et en faire un **workshop paper**. Application pure : on **réutilise l'embedder déjà entraîné** (pas de ré-entraînement), on rejoue **OPENER-Sup** et **OPENER-ZS transductif + fusion** sur des **datasets légaux**.

---

## 🎯 Vue d'ensemble

Même pipeline que le papier principal (**Detect → Embed → Type**, aucun encodeur entraîné from scratch), appliqué au domaine **juridique** :
1. **Mention Detection** : GLiNER-L (frozen, zero-shot).
2. **Embedding** : Nomic v1.5 Matryoshka **fine-tuné contrastif + hard-mining** — le **même modèle** que le papier principal, repris **depuis HuggingFace** (transfert au légal, pas de re-fine-tuning).
3. **Entity Typing**, 2 points de fonctionnement :
   - **OPENER-Sup** : LinearSVC `class_weight='balanced'` sur le train cible.
   - **OPENER-ZS** : prototypes label-name + **raffinement transductif** (spherical k-means) + **fusion** avec le label zero-shot de GLiNER (β). = la meilleure variante ZS du papier principal.

**Angle du papier** : « Un pipeline NER open-world frugal, entraîné sur du généraliste, **transfère-t-il au juridique** sans adaptation ? » → étude de transfert zero-shot/supervisé léger au domaine légal, 3 axes (AMI + latence + énergie) comme le papier principal.

---

## ✅ Décisions actées

| Sujet | Choix |
|---|---|
| Emplacement | Dossier frère `LyRIDS_Opener_Legal/` (repo séparé) |
| Portée | **Application pure** : réutiliser l'embedder existant (pas de ré-entraînement) |
| Variante ZS | **Transductif + fusion détecteur** (comme le papier principal) |
| Source embedder | **HuggingFace** ✅ `Thibault-GAREL/opener-sup` + `opener-zs` (cf. `configs/legal.yaml`) |
| Architecture / figure | **Identique** au papier principal (SVG/PDF copiés) |
| Datasets (actés) | **E-NER** (en, ancre domaine) + **LeNER-Br** (pt) + **German-LER** (de) — anglais + multilingue. `src/data/legal_datasets.py` ✅ écrit + **validé** |
| Venue | **NLLP @ EMNLP** (workshop ACL, ~8 p) → ⚠️ **template ACL** (pas IEEEtran) |

### 📚 Datasets retenus (loader validé)

| Clé | Dataset | Langue | Types | Transfert testé | Source |
|---|---|---|---|---|---|
| `e_ner` | E-NER (SEC/EDGAR), NLLP 2022 | en | 7 (BUSINESS, COURT, GOVERNMENT, LEGISLATION/ACT, LOCATION, PERSON, MISCELLANEOUS) | **domaine pur** (Nomic est EN) | GitHub `terenceau1/E-NER-Dataset` (CoNLL, split 80/20 seed 42) |
| `lener_br` | LeNER-Br, PROPOR 2018 | pt | 6 (PESSOA, ORGANIZACAO, LOCAL, TEMPO, LEGISLACAO, JURISPRUDENCIA) | cross-lingue + domaine | HF `peluz/lener_br` |
| `german_ler` / `_coarse` | German-LER, LREC 2020 | de | 18 / 7 | cross-lingue + domaine | HF `elenanereiss/german-ler` |

**Story** : E-NER isole le transfert de **domaine** (anglais→anglais légal, là où OPENER devrait briller), LeNER-Br/German-LER ajoutent le stress **cross-lingue** (où le transfert se dégrade → motive l'adaptation légère). E-NER est **lui-même publié à NLLP** et motive notre thèse (« NER général se dégrade sur le légal »).

## ⏳ Détails EN ATTENTE (à confirmer par Thibault)

1. **Template ACL** : basculer `paper/` de IEEEtran → **ACL** (`acl_natbib`/`emnlp` style). Chantier en cours.
2. **PDF des références légales** à déposer dans `paper/paper_used/` pour vérifier les venues (cf. feedback "pas de venue hallucinée") : Legal-BERT, **E-NER** (NLLP 2022, p. 246-255 ✅ déjà vérifié via ACL Anthology), LeNER-Br (PROPOR 2018), German-LER (LREC 2020), LexGLUE, LegalEval/SemEval-2023 Task 6.

---

## 🧱 Pipeline réutilisé (depuis LyRIDS_Opener)

- `src/models/embedder.py` — wrapper Nomic Matryoshka (span-in-context, `classification: ` + `[ENT]…[/ENT]`, 768d).
- `src/data/{schema,owner_datasets}.py` — format OWNER + loader (les datasets légaux seront convertis à ce format, ou un loader dédié `legal_datasets.py`).
- `src/utils/{energy,timing}.py` — mesure énergie/latence (3 axes).
- Scripts d'éval à copier/adapter depuis le repo principal :
  - `run_balanced_classifiers.py` → **OPENER-Sup** (gold) ; `run_opener_e2e.py` → Sup e2e.
  - `run_opener_zs_e2e_fusion.py` → **OPENER-ZS** transductif + fusion.
  - Adaptation : `--embedder <HF repo>` + loader légal.

## ⚠️ GPU

L'embedding des datasets légaux **nécessite le GPU** (6 Go). Tant que le **run multi-seed du papier principal tourne (~10h)**, ne pas lancer d'éval GPU ici (contention/OOM). Lancer les évals légales **après**.

## 🗺️ Plan

1. ✅ Lien HF reçu (`opener-sup`/`opener-zs`).
2. ✅ `legal_datasets.py` écrit + **validé** (E-NER en / LeNER-Br pt / German-LER de) + `configs/legal.yaml`.
3. ✅ Datasets (anglais+multilingue) + venue (NLLP) **actés**. PDF légaux déposés + triés (`paper_used/`, 5 réfs vérifiées).
4. ✅ Template ACL + Related Work légale + Datasets section (vraies distributions).
5. ✅ **Harness d'éval câblé** : 4 protocoles (`run_balanced_classifiers`/`run_opener_e2e`/`run_opener_zs`/`run_opener_zs_e2e_fusion`) avec `--legal` + embedder HF. Dico d'anchors EN (`configs/legal_anchors.yaml`). Driver `scripts/run_legal.sh`. **Validé CPU end-to-end** (Sup-gold sur E-NER).
6. ✅ **Évals lancées** (`run_legal.sh`, embedder local = poids HF) → résultats dans `outputs/results/legal/` (4 protocoles, 0 erreur).
7. ✅ **Paper rempli** avec les vrais chiffres : Abstract, Intro, Results (Table 2 transfert AMI + Table 3 frugalité + Fig 2 bar chart), Analysis, Conclusion/Limitations. **Compile propre : 7 p, 0 undef, 0 overfull.**
8. **Reste** : proofread final ; éventuellement valider le chargement HF réel de l'embedder (sans GPU) ; générer le zip Overleaf ; puis prêt workshop NLLP.

### 🧩 Baselines (façon journal) sur le légal
- **Scripts** (dans `scripts/baselines/`) : `run_gliner`, `run_gner`, `run_llm_int4` (+ `owner_export`, `owner_collect`), tous avec `--legal`. Drivers : `scripts/run_legal_baselines.sh` (GLiNER S/M/L + GNER + Qwen-int4) et `scripts/run_legal_owner.sh` (OWNER).
- ⚠️ **OWNER** : sa codebase + son checkpoint sont un **sous-module git du repo principal** (`../LyRIDS_Opener/external/OWNER`, plusieurs Go partagés avec le papier journal). On ne le duplique **pas** : `run_legal_owner.sh` le **référence** (env conda `D:\conda_envs\owner`) et écrit les logs/résultats ici. Protocole transfert (checkpoint conll2003 chargé, zéro-shot), **aucun réentraînement**.
- Résultats : `outputs/results/baselines/{gliner_*,gner,qwen1_5b,owner}/` ; logs `outputs/logs/legal_*.log`.

### 📊 Résultats clés (AMI ×100, gold)
- **OPENER-Sup** : 70.0 (en) / 53.3 (pt) / 59.1 (de) → **60.8 moy**, robuste cross-lingue.
- **OPENER-ZS** : 44.0 (en) / 31.6 (pt) / 22.6 (de) → 32.7 moy, se dégrade cross-lingue.
- **e2e** plafonné ~34-35 par le **recall du détecteur anglais** (57%→19% en→de) = goulot.
- Frugalité : ZS 0.55 Wh/31ms, Sup 1.73 Wh/22ms.

### ▶️ Commande à lancer quand le GPU est libre
```powershell
& c:\0-Code_py_temp\pytorch_cuda_env\Scripts\Activate.ps1
bash scripts/run_legal.sh   # embedder HF par défaut ; fallback local possible via $EMBEDDER
```
Le script écrit un log horodaté dans `outputs/logs/legal_run_*.log` (suivi : `Get-Content <log> -Wait -Tail 30`).

## 🔗 Lien avec le papier principal

Le papier principal (`../LyRIDS_Opener`) reste la **publication journal**. Ce projet est le **workshop applicatif légal**, indépendant (git/paper séparés), mais réutilise modèle + architecture + code.
