"""Confronte chaque reference citee a la 1re page de son PDF (titre + 1er auteur + annee)."""
import re
import subprocess
import unicodedata
from pathlib import Path

AUX = 'paper/main.aux'
BIB = 'paper/references.bib'
PDF_ROOT = Path('paper/paper_used')

# clef bib -> nom de fichier PDF (sans extension)
MAP = {
    'aly2021zeroshot': 'Leveraging Type Descriptions',
    'anon2026opener': 'OPENER - KBS paper',
    'ariai2025legalnlp': 'Natural Language Processing for the Legal Domain',
    'au2022ener': 'E-NER',
    'chalkidis2020legalbert': 'LEGAL-BERT',
    'chalkidis2022lexglue': 'LexGLUE',
    'ding2024gner': 'Rethinking Negative Instances',
    'fan2008liblinear': 'LIBLINEAR',
    'genest2025owner': 'OWNER',
    'he2023debertav3': 'DEBERTAV3',
    'henderson2020systematic': 'Towards the Systematic Reporting',
    'kalamkar2022indianlegal': 'Named Entity Recognition in Indian court',
    'kalusev2025serbian': 'Named Entity Recognition for Serbian',
    'karamitsos2025legner': 'LegNER',
    'kusupati2022matryoshka': 'Matryoshka Representation Learning',
    'lacoste2019quantifying': 'Quantifying the Carbon Emissions',
    'dettmers2023qlora': 'QLORA',
    'codecarbon2024': None,  # logiciel, pas d'article
    'leitner2020germanler': 'A Dataset of German Legal Documents',
    'lu2025financial': 'Financial Named Entity Recognition',
    'luccioni2024power': 'Power Hungry Processing',
    'luzdearaujo2018lenerbr': 'LeNER-Br',
    'nussbaum2024nomic': 'Nomic Embed',
    'patterson2022carbon': 'The Carbon Footprint of Machine Learning Training',
    'pedregosa2011scikit': 'Scikit-learn',
    'qwen2025': 'Qwen2.5 Technical Report',
    'raffel2020t5': 'Exploring the Limits of Transfer Learning',
    'ramesh2023flairnlp': 'FlairNLP at SemEval-2023',
    'reimers2019sbert': 'Sentence-BERT',
    'sainz2024gollie': 'GoLLIE',
    'schroff2015facenet': 'FaceNet',
    'schwartz2020greenai': 'Green AI',
    'strubell2019energy': 'Energy and Policy Considerations',
    'tjongkimsang2003conll': 'Introduction to the CoNLL-2003',
    'vinh2010ami': 'Information Theoretic Measures',
    'wei2024chatie': 'ChatIE',
    'wu2022sustainable': 'Sustainable AI',
    'zaratiana2024gliner': 'GLiNER',
    'zhou2024universalner': 'UniversalNER',
}


def norm(s):
    s = unicodedata.normalize('NFKD', s)
    s = ''.join(c for c in s if not unicodedata.combining(c))
    return re.sub(r'[^a-z0-9]+', ' ', s.lower()).strip()


aux = open(AUX, encoding='utf-8', errors='replace').read()
cited = set()
for u in re.findall(re.escape(chr(92) + 'citation') + r'\{([^}]*)\}', aux):
    cited |= {k.strip() for k in u.split(',')}
cited.discard('*')

bib = open(BIB, encoding='utf-8').read()
entries = {}
for block in re.split(r'\n(?=@)', bib):
    m = re.match(r'@\w+\{([^,]+),', block)
    if m:
        entries[m.group(1).strip()] = block

pdfs = list(PDF_ROOT.rglob('*.pdf'))
problems = []

for key in sorted(cited):
    e = entries.get(key, '')
    title = re.search(r'title\s*=\s*[{"](.+?)[}"],?\s*\n', e, re.S)
    title = re.sub(r'[{}]', '', title.group(1)) if title else '?'
    author = re.search(r'author\s*=\s*\{(.+?)\},?\s*\n', e, re.S)
    first = author.group(1).split(' and ')[0].strip() if author else '?'
    first_last = re.sub(r'[{}\\\'"~^]', '', first.split(',')[0]).split()[-1] if first != '?' else '?'
    year = re.search(r'year\s*=\s*[{"]?(\d{4})', e)
    year = year.group(1) if year else '?'

    if key in MAP and MAP[key] is None:
        print(f'  -- {key:28s} logiciel, pas de PDF attendu')
        continue
    frag = MAP.get(key)
    hit = [p for p in pdfs if frag and norm(frag) in norm(p.stem)] if frag else []
    if not hit:
        problems.append(f'{key}: AUCUN PDF')
        print(f'  !! {key:28s} AUCUN PDF')
        continue
    txt = subprocess.run(['pdftotext', '-f', '1', '-l', '1', str(hit[0]), '-'],
                         capture_output=True, text=True, encoding='utf-8',
                         errors='replace').stdout
    n = norm(txt)
    words = [w for w in norm(title).split() if len(w) > 3][:6]
    t_ok = sum(w in n for w in words) >= max(2, len(words) - 1)
    a_ok = norm(first_last) in n
    y_ok = year in txt or year == '?'
    flag = '' if (t_ok and a_ok) else '  <-- A VERIFIER'
    if flag:
        problems.append(f'{key}: titre={t_ok} auteur={a_ok}')
    print(f'  {"OK " if not flag else "??"} {key:28s} titre={int(t_ok)} auteur={int(a_ok)} annee={int(bool(y_ok))}  [{hit[0].parent.name}]{flag}')

print()
print('PDFs presents non rattaches a une cle citee :')
mapped = {norm(v) for v in MAP.values()}
for p in sorted(pdfs):
    if not any(m in norm(p.stem) for m in mapped):
        print('  ORPHELIN:', p.parent.name + '/' + p.name)
print()
print('%d problemes' % len(problems))
