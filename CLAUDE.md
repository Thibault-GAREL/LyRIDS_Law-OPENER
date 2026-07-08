# CLAUDE.md — OPENER-Legal (workshop paper)

Projet **frère** de `LyRIDS_Opener`. Objectif : **appliquer OPENER au NER juridique** (legal domain) et en faire un **workshop paper**. Application pure : on **réutilise l'embedder déjà entraîné** (pas de ré-entraînement), on rejoue **OPENER-Sup** et **OPENER-ZS transductif + fusion** sur des **datasets légaux**.

---

## 🎯 Vue d'ensemble

Même pipeline que le papier principal (**Detect → Embed → Type**, aucun encodeur entraîné from scratch), appliqué au domaine **juridique** :
1. **Mention Detection** : GLiNER frozen, zero-shot ; meilleur détecteur par tête (**GLiNER-M + fit-on-detected** pour Sup, **GLiNER-L** pour ZS).
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
| Datasets (actés) | **4 datasets** : **E-NER** (en) + **Indian-Legal** (en) + **LeNER-Br** (pt) + **German-LER** (de). MAPA (en+fr) et CUAD testés puis **écartés** (pas de NER légal-spécifique). `src/data/legal_datasets.py` ✅ |
| Venue | **NLLP @ EMNLP** (workshop ACL, ~8 p) → **template ACL** ✅ en place |
| Détecteurs e2e (actés) | **GLiNER-M + `--fit-on-detected`** pour OPENER-Sup, **GLiNER-L** pour OPENER-ZS (meilleur détecteur par tête, cf. Setup du paper) |
| Embedder HF | ✅ **vérifié bit à bit identique** au checkpoint local (`scripts/verify_hf_embedder.py`, 112/112 tenseurs, `opener-sup` ET `opener-zs`) |

### 📚 Datasets retenus (loader validé)

| Clé | Dataset | Langue | Types | Transfert testé | Source |
|---|---|---|---|---|---|
| `e_ner` | E-NER (SEC/EDGAR), NLLP 2022 | en | 7 (BUSINESS, COURT, GOVERNMENT, LEGISLATION/ACT, LOCATION, PERSON, MISCELLANEOUS) | **domaine pur** (Nomic est EN) | GitHub `terenceau1/E-NER-Dataset` (CoNLL, split 80/20 seed 42) |
| `indian_legal` | Indian court judgments (Kalamkar et al.), NLLP 2022 | en | 14 types légaux fins (COURT, JUDGE, LAWYER, STATUTE, PROVISION, PRECEDENT…) | domaine pur, schéma fin | HF `AjayMukundS/Indian_Legal_NER_Dataset` (spans caractères, eval sur `validation`) |
| `lener_br` | LeNER-Br, PROPOR 2018 | pt | 6 (PESSOA, ORGANIZACAO, LOCAL, TEMPO, LEGISLACAO, JURISPRUDENCIA) | cross-lingue + domaine | HF `peluz/lener_br` |
| `german_ler` / `_coarse` | German-LER, LREC 2020 | de | 19 / 7 (coarse en tables, fine en stress test Analysis) | cross-lingue + domaine | HF `elenanereiss/german-ler` |

**Story** : E-NER + Indian-Legal isolent le transfert de **domaine** (anglais→anglais légal, là où OPENER devrait briller), LeNER-Br/German-LER ajoutent le stress **cross-lingue** (où le transfert se dégrade → motive l'adaptation légère). E-NER et Indian-Legal sont **eux-mêmes publiés à NLLP** et motivent notre thèse (« NER général se dégrade sur le légal »). MAPA (en/fr) et CUAD évalués puis **écartés du paper** (anonymisation générique / clauses : pas du NER légal-spécifique).

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
6. ✅ **Évals lancées** (embedder local = poids HF, vérifié) → résultats dans `outputs/results/legal/` + `outputs/results/baselines/` (4 datasets, 0 erreur). **Chiffres finaux** = runs bootstrap du 2026-07-05 (`bootstrap_ci/` gold, `legal/bootstrap_e2e/` e2e apparié), PAS les drivers.
7. ✅ **Paper rempli** avec les vrais chiffres : Abstract, Intro, Related Work, Method, Results (Table 1 datasets, Table 2 gold, Table 3 e2e, Tables 4-5 latence/énergie), Analysis (stress 19 types, sweep β), Conclusion/Limitations. **Compile propre : 8 p (dont ~1,5 p de biblio), 0 undef, 0 overfull** (équation + tables resserrées le 2026-07-08).
8. ✅ **Embedder HF validé** : `scripts/verify_hf_embedder.py` → poids `opener-sup`/`opener-zs` identiques au checkpoint local.
9. **Reste** : proofread final ; générer le zip Overleaf ; README (à la toute fin, une fois le projet figé) ; puis prêt workshop NLLP.

### 🧩 Baselines (façon journal) sur le légal
- **Scripts** (dans `scripts/baselines/`) : `run_gliner`, `run_gner`, `run_llm_int4` (+ `owner_export`, `owner_collect`), tous avec `--legal`. Drivers : `scripts/run_legal_baselines.sh` (GLiNER S/M/L + GNER + Qwen-int4) et `scripts/run_legal_owner.sh` (OWNER).
- ⚠️ **OWNER** : sa codebase + son checkpoint sont un **sous-module git du repo principal** (`../LyRIDS_Opener/external/OWNER`, plusieurs Go partagés avec le papier journal). On ne le duplique **pas** : `run_legal_owner.sh` le **référence** (env conda `D:\conda_envs\owner`) et écrit les logs/résultats ici. Protocole transfert (checkpoint conll2003 chargé, zéro-shot), **aucun réentraînement**.
- Résultats : `outputs/results/baselines/{gliner_*,gner,qwen1_5b,owner}/` ; logs `outputs/logs/legal_*.log`.

### 📊 Résultats clés (AMI ×100, **mean±std sur 3 seeds de ré-entraînement de l'embedder** 7/42/123, 4 datasets : e_ner / indian_legal / lener_br / german_ler_coarse)
- **Gold — OPENER-Sup** : 70.9±1.2 / 60.8±1.4 / 53.7±1.2 / 60.3±1.0 → **61.4 moy**, robuste cross-lingue ET stable en seed (German fine 19 types : 61.5±0.7).
- **Gold — OPENER-ZS** : 47.7±3.5 / 49.1±3.0 / 37.4±6.0 / 23.8±4.3 → 39.5 moy. ⚠️ tête **sensible au seed** (std 3-6 ; le released seed 42 est un tirage BAS sur l'anglais), se dégrade cross-lingue (German fine : 22.7±2.1).
- **e2e** : bande étroite ~33-36 plafonnée par le **recall du détecteur anglais** (GLiNER-L 57.4%→18.7% en→de). **OPENER-Sup meilleur moy 35.9±0.4** (devant GLiNER-L 35.0 et GLiNER-M 34.9 pour CHAQUE seed : 35.5/35.9/36.2 ; égalité GLiNER-L sur Indian 35.2) ; ZS 33.4±0.4 juste sous OWNER 33.7 ; lead Sup **significatif** (bootstrap apparié released : +1.0 vs GLiNER-M CI95 [+0.4,+1.7], +1.0 vs GLiNER-L [+0.1,+1.9]).
- Frugalité e2e (mesurée sur le released) : Sup 84 ms / 1.4 Wh, ZS 133 ms / 2.2 Wh ; 1 à 2 ordres de grandeur sous GNER (8.3 s, 79 Wh) et Qwen int4.
- **Multi-seed** : évals `outputs/results/seed_runs/seed{7,123}/` (`scripts/run_multiseed.sh`, embedders `../LyRIDS_Opener/outputs/models/seed{S}_hard`, rien de ré-entraîné ici) ; agrégation `scripts/aggregate_multiseed.py` (`--check` = validation vs tables du paper) ; vérif mécanique `scripts/verify_paper_numbers.py` (cellules .tex vs JSON + chiffres dérivés + phrases seed). Convention : chiffres dérivés = arithmétique des cellules affichées.

### ▶️ Commande à lancer quand le GPU est libre
```powershell
& c:\0-Code_py_temp\pytorch_cuda_env\Scripts\Activate.ps1
bash scripts/run_legal.sh   # 4 datasets, détecteurs par tête (M/L) ; embedder HF par défaut, fallback via $EMBEDDER
```
Le script écrit un log horodaté dans `outputs/logs/legal_run_*.log` (suivi : `Get-Content <log> -Wait -Tail 30`).
⚠️ Les **chiffres du paper** (points + incertitudes) viennent de `scripts/run_bootstrap_ci.py` (gold) et `scripts/run_bootstrap_e2e.py` (e2e apparié) ; les drivers ne servent qu'à reproduire les points de fonctionnement.

## 🔗 Lien avec le papier principal

Le papier principal (`../LyRIDS_Opener`) reste la **publication journal**. Ce projet est le **workshop applicatif légal**, indépendant (git/paper séparés), mais réutilise modèle + architecture + code.
