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
| Datasets (proposés) | **LeNER-Br** (pt) + **German-LER** (de) — multilingue, transfert cross-lingue. `src/data/legal_datasets.py` ✅ écrit |

## ⏳ Détails EN ATTENTE (à confirmer par Thibault)

1. **Confirmer/ajuster les datasets** : set proposé = **LeNER-Br** (pt, PROPOR 2018) + **German-LER** (de, LREC 2020), tous deux publics + BIO sur le HF Hub. ⚠️ **Conséquence majeure** : Nomic v1.5 est **anglo-centré** → c'est un transfert **cross-lingue + domaine** (plus dur, mais honnête et intéressant ; un échec partiel motive l'adaptation légère = la conclusion). Alternative anglais-pur : **E-NER** (SEC/contrats) — source HF propre à trouver. Choix = mono-anglais (isole le domaine) vs multilingue (story plus large).
2. **Workshop / venue** visé (template + limite de pages : workshop = 4-8 p ; ex. NLLP@EMNLP, JURIX, ACL/NeurIPS workshop).
3. **PDF des références légales** à déposer dans `paper/paper_used/` pour vérifier les venues (cf. feedback "pas de venue hallucinée") : Legal-BERT, LeNER-Br, German-LER, (E-NER), LexGLUE, LegalEval/SemEval-2023 Task 6.

→ Ces infos branchent : la **config** (`configs/legal.yaml`, set proposé déjà rempli), la **Related Work légale** (refs + venues, en attente PDF), et les sections **Datasets/Results** du paper.

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
2. ✅ `legal_datasets.py` écrit (LeNER-Br + German-LER, réutilise `_bio_to_spans`) + `configs/legal.yaml` rempli (set proposé).
3. **[à confirmer]** Datasets + venue (cf. EN ATTENTE) ; déposer les PDF légaux.
4. Adapter les scripts d'éval pour charger l'embedder **depuis HF** + le loader légal (`load_legal_dataset`).
5. **[après multi-seed]** Lancer Sup + ZS-fusion sur les datasets légaux (gold + e2e), 3 axes.
6. Remplir Related Work légale + Datasets/Results du paper + tables/figures (style du papier principal).
7. Compiler, proofread, prêt workshop.

## 🔗 Lien avec le papier principal

Le papier principal (`../LyRIDS_Opener`) reste la **publication journal**. Ce projet est le **workshop applicatif légal**, indépendant (git/paper séparés), mais réutilise modèle + architecture + code.
