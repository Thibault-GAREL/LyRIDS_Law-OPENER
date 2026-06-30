"""Loader des datasets NER juridiques pour OPENER-Legal.

Même contrat de sortie que ``owner_datasets.load_owner_dataset`` :
    list[(text, [(start_char, end_char, label), ...])]

On réutilise ``_bio_to_spans`` (conversion BIO → spans caractères) pour rester
strictement compatible avec la pipeline du papier principal. La seule
spécificité juridique est le *registre* de datasets ci-dessous.

Datasets candidats (publics, NER au format BIO accessibles via Parquet sur le
HF Hub). Le choix exact / la langue ciblée restent à confirmer avec Thibault :

  - ``lener_br``    : LeNER-Br, portugais, décisions de justice brésiliennes.
                      6 types (PESSOA, ORGANIZACAO, LOCAL, TEMPO, LEGISLACAO,
                      JURISPRUDENCIA).
  - ``german_ler``  : German-LER, allemand, décisions de justice fédérales.
                      Schéma *fine* (19 types) ; ``ner_coarse_tags`` donne le
                      schéma *coarse* (7 types) -> variante ``german_ler_coarse``.
  - ``e_ner``       : E-NER, anglais, dépôts SEC / contrats. Source à confirmer
                      (peut ne pas exposer de Parquet ClassLabel propre).

Comme dans ``owner_datasets``, on force ``revision='refs/convert/parquet'`` pour
contourner les *loading scripts* (supprimés dans ``datasets>=3``) et lire la
version Parquet auto-convertie du Hub.
"""
from typing import Optional

from src.data.owner_datasets import _bio_to_spans, collect_label_set  # noqa: F401


# Mapping dataset juridique -> spec de chargement.
_LEGAL_SPECS: dict[str, dict] = {
    # ----- LeNER-Br (portugais, PROPOR 2018) -----
    'lener_br': {
        'hf': 'lener_br',
        'token_col': 'tokens',
        'tag_col': 'ner_tags',
        'source': 'classlabel',
        'lang': 'pt',
    },
    # ----- German-LER (allemand, LREC 2020) : schéma fin (19 types) -----
    'german_ler': {
        'hf': 'elenanereiss/german-ler',
        'token_col': 'tokens',
        'tag_col': 'ner_tags',
        'source': 'classlabel',
        'lang': 'de',
    },
    # ----- German-LER : schéma grossier (7 types) -----
    'german_ler_coarse': {
        'hf': 'elenanereiss/german-ler',
        'token_col': 'tokens',
        'tag_col': 'ner_coarse_tags',
        'source': 'classlabel',
        'lang': 'de',
    },
    # ----- E-NER (anglais, NLLP@EMNLP 2022) : source à confirmer -----
    # 'e_ner': {
    #     'hf': '<TODO: dépôt HF exposant tokens + BIO>',
    #     'token_col': 'tokens',
    #     'tag_col': 'ner_tags',
    #     'source': 'classlabel',
    #     'lang': 'en',
    # },
}


def list_supported_legal_datasets() -> list[str]:
    return sorted(_LEGAL_SPECS.keys())


def dataset_language(name: str) -> str:
    """Langue ISO-639-1 du dataset (utile pour l'analyse cross-lingue)."""
    return _LEGAL_SPECS[name].get('lang', 'unknown')


def load_legal_dataset(
    name: str,
    split: str = 'test',
    max_sentences: Optional[int] = None,
) -> list[tuple[str, list[tuple[int, int, str]]]]:
    """Charge un dataset NER juridique au format Opener.

    Args:
        name          : voir ``list_supported_legal_datasets()``.
        split         : 'train' / 'validation' / 'test'.
        max_sentences : si fourni, tronque (les phrases sans entité ne comptent pas).

    Returns:
        Liste de ``(text, spans)``. Les phrases sans entité sont skippées,
        comme dans ``owner_datasets`` (le typing s'évalue sur des mentions).
    """
    if name not in _LEGAL_SPECS:
        raise KeyError(
            f"Dataset juridique {name!r} non supporté. "
            f"Voir list_supported_legal_datasets()."
        )
    spec = _LEGAL_SPECS[name]

    from datasets import load_dataset

    ds = load_dataset(spec['hf'], revision='refs/convert/parquet', split=split)

    if spec['source'] == 'classlabel':
        tag_feature = ds.features[spec['tag_col']]
        inner = getattr(tag_feature, 'feature', tag_feature)
        label_names = inner.names
    else:
        label_names = None

    out: list[tuple[str, list[tuple[int, int, str]]]] = []
    for ex in ds:
        tokens = ex[spec['token_col']]
        raw_tags = ex[spec['tag_col']]
        if label_names is not None:
            tags = [label_names[t] if isinstance(t, int) else t for t in raw_tags]
        else:
            tags = list(raw_tags)
        spans = _bio_to_spans(tokens, tags)
        if not spans:
            continue
        text = ' '.join(tokens)
        out.append((text, spans))
        if max_sentences is not None and len(out) >= max_sentences:
            break
    return out
