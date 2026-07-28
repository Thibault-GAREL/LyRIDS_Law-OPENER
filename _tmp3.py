import re
aux = open('paper/main.aux', encoding='utf-8', errors='replace').read()
used = set()
for u in re.findall(re.escape(chr(92) + 'citation') + r'\{([^}]*)\}', aux):
    used |= {k.strip() for k in u.split(',')}
bib = open('paper/references.bib', encoding='utf-8').read()
keys = set(re.findall(r'@\w+\{([^,]+),', bib))
print('citees:', len(used))
print('non citees (%d):' % len(keys - used), sorted(keys - used))
print('manquantes:', sorted(used - keys))
