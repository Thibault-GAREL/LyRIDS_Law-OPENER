import re
bib = open('paper/references.bib', encoding='utf-8').read()
want = ['anon2026opener','au2022ener','kalamkar2022indianlegal','luzdearaujo2018lenerbr','leitner2020germanler','lacoste2019codecarbon','qwen2025','ariai2025legalnlp','karamitsos2025legner','kalusev2025serbian','ramesh2023flairnlp','lu2025financial','chalkidis2020legalbert','chalkidis2022lexglue','genest2025owner','aly2021zeroshot','vinh2010ami']
entries = re.split(r'\n(?=@)', bib)
for e in entries:
    m = re.match(r'@\w+\{([^,]+),', e)
    if m and m.group(1).strip() in want:
        print(e.strip()[:700]); print('-'*60)
