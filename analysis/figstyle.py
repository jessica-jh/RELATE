"""Shared typography and output settings for the paper figures.

Every figure in the paper should agree on its type: a reader flipping between
Figure 2 and Figure 3 should not see the tick labels change size. Import this
and call apply() as the first thing any figure script does.

    import figstyle
    figstyle.apply()

The size tiers below are the ones analysis/paper_figures/fig_persona.pdf uses, and they
are deliberately ranked -- title > axis label > tick/legend > inline annotation
-- so the hierarchy of the figure is legible before any of the words are read.
A single flat size for everything (e.g. 9.5pt for ticks and titles alike) reads
as unstyled no matter how good the underlying plot is.

Sizes are in points against ICLR's 10pt body. ANNOT is the floor: nothing in a
paper figure should be smaller, and it is only for text sitting directly against
the thing it names.
"""
import matplotlib

TITLE = 8.5
PANEL_TITLE = 7.5   # per-panel titles in small multiples, where 8.5 would crowd
LABEL = 8.0
TICK = 7.0
LEGEND = 7.0
ANNOT = 6.5

# Colour rule for text, so the same decision is not re-litigated per figure:
#
#   TEXT   black. Everything structural -- tick labels, axis labels, panel
#          titles, legend entries, and any annotation whose meaning is carried
#          by the words rather than by which series it belongs to.
#   MUTED  grey. Secondary/meta text that should recede: sample counts, notes
#          about a test that is reported elsewhere.
#   series colour  reserved for text that *identifies* a series -- a direct
#          label at the end of a line. Colour there is not decoration, it is
#          the only thing tying the label to its line, so those stay coloured.
#
# The practical upshot: if you could delete the colour and still know what the
# text refers to, it should be black.
TEXT = "black"
MUTED = "#666666"

# ICLR \textwidth in inches. Figures are saved at exactly this width so that
# \includegraphics{...} with no width= lands at scale 1 and panels line up
# across figures.
TEXTWIDTH = 5.5

RCPARAMS = {
    # Times-compatible serif, matching \usepackage{times} in relate.tex. STIX
    # covers the math glyphs (÷, Δ, −) that DejaVu Serif renders off-weight.
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "DejaVu Serif", "Times New Roman"],
    "mathtext.fontset": "stix",
    "font.size": LABEL,
    "axes.titlesize": TITLE,
    "axes.labelsize": LABEL,
    "legend.fontsize": LEGEND,
    "xtick.labelsize": TICK,
    "ytick.labelsize": TICK,

    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "lines.linewidth": 1.1,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.22,
    "grid.linewidth": 0.4,
    "hatch.linewidth": 0.35,

    # Type 42 (TrueType), not matplotlib's Type 3 default: arXiv warns on Type 3
    # and it renders poorly when a reviewer zooms.
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    # No tight-bbox cropping -- it crops to the ink, so the saved width depends
    # on how far the tick labels happen to stick out and every figure ends up a
    # slightly different size. constrained_layout packs inside a fixed canvas.
    "savefig.bbox": "standard",
    "savefig.pad_inches": 0.0,
    "figure.dpi": 400,
    "figure.constrained_layout.use": True,
    "figure.constrained_layout.h_pad": 0.02,
    "figure.constrained_layout.w_pad": 0.02,
    "figure.constrained_layout.hspace": 0.03,
    "figure.constrained_layout.wspace": 0.03,
}


def apply():
    matplotlib.rcParams.update(RCPARAMS)


def luminance(hexc):
    """WCAG relative luminance of an #RRGGBB colour."""
    ch = [int(hexc.lstrip("#")[i:i + 2], 16) / 255 for i in (0, 2, 4)]
    ch = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in ch]
    return 0.2126 * ch[0] + 0.7152 * ch[1] + 0.0722 * ch[2]


def contrast_on_white(hexc):
    return 1.05 / (luminance(hexc) + 0.05)


def darken_for_text(hexc, target=3.0):
    """Darken a colour until it clears `target`:1 contrast against white.

    A hue that reads fine as a 1.2pt line can disappear as 6.5pt text -- there is
    far less ink per glyph than per line. This keeps the hue (so a direct label
    still visibly belongs to its line) while making the text dark enough to read,
    rather than maintaining a second hand-picked palette for label text.
    """
    r, g, b = [int(hexc.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4)]
    for _ in range(80):
        cur = "#%02X%02X%02X" % (r, g, b)
        if contrast_on_white(cur) >= target:
            return cur
        r, g, b = (max(0, int(v * 0.95)) for v in (r, g, b))
    return "#%02X%02X%02X" % (r, g, b)
