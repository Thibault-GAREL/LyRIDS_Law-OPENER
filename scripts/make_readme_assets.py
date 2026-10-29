"""Génère les visuels du README (figures de résultats + pages du papier).

Deux figures, chacune en version claire et sombre (balise <picture> du README) :
  - legal-fig-gold   : AMI de typage sur mentions gold, par corpus (tab:legal-gold).
  - legal-fig-cost   : AMI end-to-end contre énergie, moyenne des 4 corpus (tab:legal-efficiency).
Puis la miniature cliquable du talk NLLP (legal-talk-thumbnail, clair + sombre, lien YouTube
dans le README, GitHub n'autorise pas d'iframe) et les 15 pages du PDF camera-ready en PNG
(assets/paper_pages/).

Les chiffres sont recopiés des tables du paper (paper/sections/04_experiments.tex),
elles-mêmes vérifiées par scripts/verify_paper_numbers.py. Aucun calcul ici.

Usage (venv basic_env, matplotlib + PyMuPDF, pas de GPU) :
    python -m scripts.make_readme_assets
"""
from pathlib import Path

import fitz  # PyMuPDF
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.path import Path as MplPath
from matplotlib.patches import FancyBboxPatch, PathPatch, Polygon, Rectangle

OUT_DIR = Path('assets/readme')
PAGES_DIR = Path('assets/paper_pages')
PAPER_PDF = Path('paper/OPENER-Legal_NLLP2026_camera-ready.pdf')

# Palette validée (skill dataviz, scripts/validate_palette.js, light + dark, all-pairs).
THEMES = {
    'light': dict(surface='#fcfcfb', ink='#0b0b0b', ink2='#52514e', muted='#898781',
                  grid='#e1e0d9', axis='#c3c2b7',
                  sup='#2a78d6', zs='#eb6834', owner='#1baf7a', other='#898781'),
    'dark': dict(surface='#1a1a19', ink='#ffffff', ink2='#c3c2b7', muted='#898781',
                 grid='#2c2c2a', axis='#383835',
                 sup='#3987e5', zs='#d95926', owner='#199e70', other='#898781'),
}

# tab:legal-gold (typing on gold, AMI x100). OPENER = mean ± std sur 3 seeds d'embedder.
CORPORA = ['E-NER\n(en)', 'Indian-Legal\n(en)', 'LeNER-Br\n(pt)', 'German-LER\n(de)', 'Mean']
GOLD = {
    'OWNER':      ([54.5, 59.0, 39.7, 25.7, 44.7], None),
    'OPENER-ZS':  ([47.7, 49.1, 37.4, 23.8, 39.5], [3.5, 3.0, 6.0, 4.3, 3.0]),
    'OPENER-Sup': ([70.9, 60.8, 53.7, 60.3, 61.4], [1.2, 1.4, 1.2, 1.0, 0.4]),
}

# tab:legal-efficiency (end-to-end, moyenne des 4 corpus) : (AMI x100, Wh pour 1000 phrases, clé couleur).
COST = {
    'GLiNER-S':     (33.0, 0.5, 'other'),
    'GLiNER-M':     (34.9, 0.7, 'other'),
    'GLiNER-L':     (35.0, 1.6, 'other'),
    'GNER-T5':      (32.4, 96.0, 'other'),
    'Qwen2.5-1.5B': (23.1, 100.5, 'other'),
    'OWNER':        (33.7, 14.6, 'owner'),
    'OPENER-ZS':    (33.4, 2.7, 'zs'),
    'OPENER-Sup':   (35.9, 1.7, 'sup'),
}
# Placement des étiquettes directes (décalage en points, alignement horizontal).
LABEL_POS = {
    'GLiNER-S': ((9, -4), 'left'), 'GLiNER-M': ((-9, 5), 'right'),
    'GLiNER-L': ((9, -12), 'left'), 'GNER-T5': ((-9, -4), 'right'),
    'Qwen2.5-1.5B': ((-9, -4), 'right'), 'OWNER': ((9, -4), 'left'),
    'OPENER-ZS': ((9, -4), 'left'), 'OPENER-Sup': ((-9, 6), 'right'),
}

DPI_LAYOUT = 100   # 1 px de mise en page = 1 px CSS à 900 px d'affichage
DPI_SAVE = 200     # rendu net sur écran haute densité


def _set_font():
    names = {f.name for f in font_manager.fontManager.ttflist}
    # DejaVu Sans en repli : Segoe UI n'a pas la flèche « ↖ ».
    family = ['Segoe UI', 'DejaVu Sans'] if 'Segoe UI' in names else ['DejaVu Sans']
    plt.rcParams.update({'font.family': family, 'font.size': 10})


def _px_to_data(ax, px):
    """Convertit une longueur en pixels en unités de données (x, y)."""
    (x0, y0), (x1, y1) = ax.transData.transform([(0, 0), (1, 1)])
    return px / abs(x1 - x0), px / abs(y1 - y0)


def _rounded_bar(ax, x_left, width, height, color, radius_px=4):
    """Barre carrée à la base, arrondie de 4 px au sommet (spec dataviz)."""
    rx, ry = _px_to_data(ax, radius_px)
    rx, ry = min(rx, width / 2), min(ry, height)
    xl, xr, h, k = x_left, x_left + width, height, 0.5523
    verts = [(xl, 0), (xl, h - ry),
             (xl, h - ry + k * ry), (xl + rx - k * rx, h), (xl + rx, h),
             (xr - rx, h),
             (xr - rx + k * rx, h), (xr, h - ry + k * ry), (xr, h - ry),
             (xr, 0), (xl, 0)]
    codes = [MplPath.MOVETO, MplPath.LINETO,
             MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
             MplPath.LINETO,
             MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
             MplPath.LINETO, MplPath.CLOSEPOLY]
    ax.add_patch(PathPatch(MplPath(verts, codes), facecolor=color, edgecolor='none'))


def _header(fig, t, title, subtitle, legend):
    """Titre, sous-titre et légende en ligne, alignés à gauche au-dessus du tracé."""
    fig.text(0.075, 0.95, title, fontsize=14, fontweight='semibold', color=t['ink'], va='top')
    fig.text(0.075, 0.885, subtitle, fontsize=10, color=t['ink2'], va='top')
    renderer = fig.canvas.get_renderer()
    x = 0.075
    for name, color in legend:
        fig.patches.append(plt.Rectangle((x, 0.808), 0.012, 0.024, transform=fig.transFigure,
                                         facecolor=color, edgecolor='none'))
        label = fig.text(x + 0.018, 0.82, name, fontsize=10, color=t['ink'], va='center')
        # Item suivant placé après la largeur réellement rendue du texte.
        x = label.get_window_extent(renderer).x1 / fig.bbox.width + 0.03


def _style_axes(ax, t):
    ax.set_facecolor(t['surface'])
    for side in ('top', 'right', 'left'):
        ax.spines[side].set_visible(False)
    ax.spines['bottom'].set_color(t['axis'])
    ax.tick_params(colors=t['muted'], length=0, labelsize=9.5)
    ax.yaxis.grid(True, color=t['grid'], linewidth=1, linestyle='-')
    ax.set_axisbelow(True)


def make_gold(theme):
    t = THEMES[theme]
    fig = plt.figure(figsize=(9, 4.6), dpi=DPI_LAYOUT, facecolor=t['surface'])
    ax = fig.add_axes([0.075, 0.14, 0.9, 0.6])
    _style_axes(ax, t)
    ax.set_xlim(-0.6, len(CORPORA) - 0.4)
    ax.set_ylim(0, 85)
    ax.set_yticks([0, 20, 40, 60, 80])

    bar_w, gap = _px_to_data(ax, 24)[0], _px_to_data(ax, 4)[0]
    systems = [('OWNER', t['owner']), ('OPENER-ZS', t['zs']), ('OPENER-Sup', t['sup'])]
    group_w = 3 * bar_w + 2 * gap
    for i, (name, color) in enumerate(systems):
        values, stds = GOLD[name]
        for j, v in enumerate(values):
            x_left = j - group_w / 2 + i * (bar_w + gap)
            _rounded_bar(ax, x_left, bar_w, v, color)
            top = v
            if stds is not None:
                xc = x_left + bar_w / 2
                ax.plot([xc, xc], [v - stds[j], v + stds[j]], color=t['ink2'], linewidth=1)
                ax.plot([xc - bar_w / 4, xc + bar_w / 4], [v + stds[j]] * 2, color=t['ink2'], linewidth=1)
                top = v + stds[j]
            if name == 'OPENER-Sup':
                ax.text(x_left + bar_w / 2, top + 1.5, f'{v:.1f}', ha='center', va='bottom',
                        fontsize=9.5, fontweight='semibold', color=t['ink'])

    # Le groupe « Mean » est séparé des corpus par un filet vertical discret.
    ax.axvline(len(CORPORA) - 1.5, color=t['grid'], linewidth=1)
    ax.set_xticks(range(len(CORPORA)))
    ax.set_xticklabels(CORPORA, color=t['ink2'])
    ax.set_ylabel('AMI (×100)', color=t['ink2'], fontsize=9.5)

    _header(fig, t, 'Typing on gold mentions',
            'Detection bypassed. OPENER bars are the mean over 3 embedder seeds, whiskers show ± std.',
            [(name, color) for name, color in systems])
    return fig


def make_cost(theme):
    t = THEMES[theme]
    fig = plt.figure(figsize=(9, 4.8), dpi=DPI_LAYOUT, facecolor=t['surface'])
    ax = fig.add_axes([0.075, 0.15, 0.9, 0.6])
    _style_axes(ax, t)
    ax.set_xscale('log')
    ax.set_xlim(0.3, 220)
    ax.set_ylim(21, 38)
    ticks = [0.5, 1, 2, 5, 10, 20, 50, 100]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f'{v:g}' for v in ticks])
    ax.minorticks_off()
    ax.set_xlabel('Energy per 1,000 sentences (Wh, log scale), lower is better', color=t['ink2'], fontsize=9.5)
    ax.set_ylabel('End-to-end AMI (×100)', color=t['ink2'], fontsize=9.5)

    # Diamètre 10 px, anneau de 2 px couleur surface (spec dataviz).
    marker_pt = 10 * 72 / DPI_LAYOUT
    ring_pt = 2 * 72 / DPI_LAYOUT
    order = sorted(COST, key=lambda n: COST[n][2] != 'other')  # OPENER et OWNER dessinés au-dessus
    for name in order:
        ami, wh, key = COST[name]
        ax.scatter([wh], [ami], s=marker_pt ** 2, color=t[key], edgecolors=t['surface'],
                   linewidths=ring_pt, zorder=3)
        (dx, dy), ha = LABEL_POS[name]
        ax.annotate(name, (wh, ami), xytext=(dx, dy), textcoords='offset points', ha=ha,
                    fontsize=10, color=t['ink'] if key != 'other' else t['ink2'],
                    fontweight='semibold' if name == 'OPENER-Sup' else 'normal')
    ax.text(0.33, 37.2, '↖ better', fontsize=9.5, color=t['muted'], va='center')

    _header(fig, t, 'Quality against energy, end-to-end',
            'Mean over the four legal corpora. Only OPENER-Sup uses target labels.',
            [('OPENER-Sup', t['sup']), ('OPENER-ZS', t['zs']), ('OWNER', t['owner']),
             ('Other baselines', t['other'])])
    return fig


def _play_button(fig, cx, cy, width_px):
    """Bouton play façon YouTube (rectangle rouge arrondi, triangle blanc), centré en (cx, cy)."""
    w = width_px / (fig.get_figwidth() * DPI_LAYOUT)
    h = width_px / 1.42 / (fig.get_figheight() * DPI_LAYOUT)
    ax = fig.add_axes([cx - w / 2, cy - h / 2, w, h])
    ax.set_xlim(0, 1.42)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.add_patch(FancyBboxPatch((0, 0), 1.42, 1, boxstyle='round,pad=0,rounding_size=0.28',
                                facecolor='#ff0033', edgecolor='none'))
    ax.add_patch(Polygon([(0.56, 0.3), (0.56, 0.7), (0.92, 0.5)], facecolor='white', edgecolor='none'))


def make_talk_thumbnail(theme):
    t = THEMES[theme]
    fig = plt.figure(figsize=(9, 9 * 9 / 16), dpi=DPI_LAYOUT, facecolor=t['surface'])

    fig.text(0.06, 0.86, 'NLLP 2026 @ EMNLP  ·  LIVE TALK', fontsize=10.5, fontweight='semibold',
             color=t['sup'], va='top')
    fig.text(0.06, 0.77, 'OPENER: A Frugal\nOpen-World NER Pipeline\nfor the Legal Domain',
             fontsize=23, fontweight='semibold', color=t['ink'], va='top', linespacing=1.25)
    fig.text(0.06, 0.36, 'Thibault Garel, Faiza Belbachir, Assia Soukane', fontsize=11.5,
             color=t['ink2'], va='top')
    fig.text(0.06, 0.29, 'LyRIDS, ECE Paris', fontsize=11.5, color=t['muted'], va='top')
    _play_button(fig, 0.06 + 0.024, 0.14, 30)
    fig.text(0.06 + 0.06, 0.14, 'Watch on YouTube', fontsize=12.5, fontweight='semibold',
             color=t['ink'], va='center')

    # Page 1 du papier à droite, recadrée au ratio du cadre, bouton play par-dessus.
    page = plt.imread(PAGES_DIR / 'page-01.png')
    box = [0.6, 0.08, 0.34, 0.84]
    ratio = (box[2] * fig.get_figwidth()) / (box[3] * fig.get_figheight())
    ph, pw = page.shape[:2]
    crop_h = min(ph, int(pw / ratio))
    crop_w = int(crop_h * ratio)
    x0 = (pw - crop_w) // 2
    ax = fig.add_axes(box)
    ax.imshow(page[:crop_h, x0:x0 + crop_w], interpolation='lanczos')
    ax.axis('off')
    ax.add_patch(Rectangle((0, 0), 1, 1, transform=ax.transAxes, fill=False,
                           edgecolor=t['axis'], linewidth=1))
    _play_button(fig, box[0] + box[2] / 2, box[1] + box[3] / 2, 96)
    return fig


def render_paper_pages():
    PAGES_DIR.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(PAPER_PDF)
    for i, page in enumerate(doc, start=1):
        # 130 dpi : même largeur (1105 px) que les pages du README du repo principal.
        page.get_pixmap(dpi=130).save(PAGES_DIR / f'page-{i:02d}.png')
    print(f'{doc.page_count} pages -> {PAGES_DIR}')


def main():
    _set_font()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for theme in THEMES:
        for stem, maker in (('legal-fig-gold', make_gold), ('legal-fig-cost', make_cost),
                            ('legal-talk-thumbnail', make_talk_thumbnail)):
            fig = maker(theme)
            path = OUT_DIR / f'{stem}-{theme}.png'
            fig.savefig(path, dpi=DPI_SAVE, facecolor=fig.get_facecolor())
            plt.close(fig)
            print(path)
    render_paper_pages()


if __name__ == '__main__':
    main()
