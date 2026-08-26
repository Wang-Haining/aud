"""
nature_style.py -- Matplotlib/Seaborn configuration for Nature Portfolio journals.

Origin: the unmodified helper from the nature-visualizer authoring skill, kept here
because the producers that predate npj_style.py still import from it. It is not this
project's work; npj_style.py is, and new producers should use that instead.

Usage:
    from nature_style import apply_nature_style, nature_figsize, WONG, add_panel_labels
    apply_nature_style()
    fig, ax = plt.subplots(figsize=nature_figsize('single'))
    ...
    add_panel_labels([ax])
    fig.savefig('figure1.pdf')
"""

import matplotlib as mpl
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Wong / Okabe-Ito palette  (Nature Methods 8, 441, 2011)
# The gold-standard colorblind-safe categorical palette, explicitly
# recommended by Nature Portfolio for all scientific figures.
# ---------------------------------------------------------------------------
WONG = [
    '#000000',  # black
    '#E69F00',  # orange
    '#56B4E9',  # sky blue
    '#009E73',  # bluish green
    '#F0E442',  # yellow
    '#0072B2',  # blue
    '#D55E00',  # vermillion
    '#CC79A7',  # reddish purple
]

# Convenient two-group subset (blue vs vermillion) -- most common in clinical
WONG_TWO = ['#0072B2', '#D55E00']

# Convenient three-group subset
WONG_THREE = ['#0072B2', '#D55E00', '#009E73']

# ---------------------------------------------------------------------------
# Figure width helpers (mm -> inches)
# ---------------------------------------------------------------------------
_WIDTHS_MM = {
    'single': 89,
    'one-half': 120,
    '1.5': 120,
    'double': 183,
    'full': 183,
}
_MAX_HEIGHT_MM = 247
_MM_TO_IN = 1 / 25.4


def nature_figsize(columns='single', height_ratio=0.75, height_mm=None):
    """Return (width_in, height_in) for a Nature Portfolio figure.

    Parameters
    ----------
    columns : str
        'single' (89 mm), 'one-half' (120 mm), or 'double' (183 mm).
    height_ratio : float
        height = width * height_ratio. Ignored if height_mm is given.
    height_mm : float, optional
        Explicit height in mm. Capped at 247 mm.

    Returns
    -------
    (float, float) -- width and height in inches.
    """
    w_mm = _WIDTHS_MM.get(columns, 89)
    if height_mm is not None:
        h_mm = min(height_mm, _MAX_HEIGHT_MM)
    else:
        h_mm = w_mm * height_ratio
        h_mm = min(h_mm, _MAX_HEIGHT_MM)
    return (w_mm * _MM_TO_IN, h_mm * _MM_TO_IN)


def nature_figsize_panels(columns='double', nrows=1, ncols=2,
                          panel_aspect=1.0, hspace_mm=4, wspace_mm=4):
    """Compute figure size for a multi-panel grid.

    Parameters
    ----------
    columns : str
        Total figure width class.
    nrows, ncols : int
        Grid dimensions.
    panel_aspect : float
        Height / width of each panel.
    hspace_mm, wspace_mm : float
        Spacing between panels in mm.

    Returns
    -------
    (float, float) -- total figure width and height in inches.
    """
    total_w_mm = _WIDTHS_MM.get(columns, 183)
    panel_w_mm = (total_w_mm - wspace_mm * (ncols - 1)) / ncols
    panel_h_mm = panel_w_mm * panel_aspect
    total_h_mm = panel_h_mm * nrows + hspace_mm * (nrows - 1)
    total_h_mm = min(total_h_mm, _MAX_HEIGHT_MM)
    return (total_w_mm * _MM_TO_IN, total_h_mm * _MM_TO_IN)


# ---------------------------------------------------------------------------
# Apply Nature style to matplotlib rcParams
# ---------------------------------------------------------------------------
def apply_nature_style(palette='wong'):
    """Set all rcParams to Nature Portfolio defaults.

    Call this once at the top of your script before creating any figures.
    """
    # Font embedding: TrueType (type 42) so text stays editable in
    # Illustrator and Nature's production pipeline can modify it.
    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['ps.fonttype'] = 42

    # Font: sans-serif, Arial preferred. Nature rejects serif fonts in figures.
    mpl.rcParams['font.family'] = 'sans-serif'
    mpl.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
    mpl.rcParams['font.size'] = 7            # Nature max for body text is 7 pt
    mpl.rcParams['axes.labelsize'] = 7
    mpl.rcParams['axes.titlesize'] = 7
    mpl.rcParams['xtick.labelsize'] = 6
    mpl.rcParams['ytick.labelsize'] = 6
    mpl.rcParams['legend.fontsize'] = 6
    mpl.rcParams['legend.title_fontsize'] = 7

    # Axes and ticks
    mpl.rcParams['axes.linewidth'] = 0.5
    mpl.rcParams['xtick.major.width'] = 0.5
    mpl.rcParams['ytick.major.width'] = 0.5
    mpl.rcParams['xtick.major.size'] = 3
    mpl.rcParams['ytick.major.size'] = 3
    mpl.rcParams['xtick.minor.width'] = 0.3
    mpl.rcParams['ytick.minor.width'] = 0.3
    mpl.rcParams['xtick.minor.size'] = 1.5
    mpl.rcParams['ytick.minor.size'] = 1.5
    mpl.rcParams['xtick.direction'] = 'out'
    mpl.rcParams['ytick.direction'] = 'out'

    # Lines
    mpl.rcParams['lines.linewidth'] = 1.0
    mpl.rcParams['lines.markersize'] = 4

    # Legend: Nature says no frames, no colored text
    mpl.rcParams['legend.frameon'] = False
    mpl.rcParams['legend.borderpad'] = 0.3

    # Grid: Nature explicitly prohibits background gridlines
    mpl.rcParams['axes.grid'] = False

    # Spines: left + bottom only (cleaner, widely used in Nature papers)
    mpl.rcParams['axes.spines.top'] = False
    mpl.rcParams['axes.spines.right'] = False

    # Figure defaults
    mpl.rcParams['figure.facecolor'] = 'white'
    mpl.rcParams['savefig.facecolor'] = 'white'
    mpl.rcParams['savefig.dpi'] = 300
    mpl.rcParams['savefig.bbox'] = 'tight'
    mpl.rcParams['savefig.pad_inches'] = 0.02
    mpl.rcParams['figure.constrained_layout.use'] = True

    # Color cycle
    if palette == 'wong':
        mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=WONG)
    elif palette == 'tableau-colorblind10':
        plt.style.use('tableau-colorblind10')


# ---------------------------------------------------------------------------
# Panel labels
# ---------------------------------------------------------------------------
def add_panel_labels(axes, x=-0.12, y=1.05, fontsize=8, start='a'):
    """Add Nature-style panel labels (bold lowercase a, b, c...).

    Parameters
    ----------
    axes : list of matplotlib Axes
        Axes to label, in reading order (left to right, top to bottom).
    x, y : float
        Label position in axes-fraction coordinates.
    fontsize : int
        Must be 8 for Nature.
    start : str
        Starting letter (default 'a').
    """
    for i, ax in enumerate(axes):
        label = chr(ord(start) + i)
        ax.text(x, y, label, transform=ax.transAxes,
                fontsize=fontsize, fontweight='bold',
                va='top', ha='right',
                family='sans-serif')


# ---------------------------------------------------------------------------
# Significance brackets (for bar charts)
# ---------------------------------------------------------------------------
def add_significance_bracket(ax, x1, x2, y, p_value, dh=0.02, fontsize=6):
    """Draw a significance bracket between two x positions.

    Parameters
    ----------
    ax : matplotlib Axes
    x1, x2 : float
        x positions of the two groups.
    y : float
        y position of the bracket (data coordinates).
    p_value : float
        P value to display. Converted to stars or exact value.
    dh : float
        Height of the bracket ends as fraction of y.
    fontsize : int
        Font size for the annotation.
    """
    # Format p value
    if p_value < 0.001:
        text = '***'
    elif p_value < 0.01:
        text = '**'
    elif p_value < 0.05:
        text = '*'
    else:
        text = 'n.s.'

    h = y * dh
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=0.5, c='black')
    ax.text((x1 + x2) / 2, y + h, text, ha='center', va='bottom',
            fontsize=fontsize)


# ---------------------------------------------------------------------------
# Export helpers
# ---------------------------------------------------------------------------
def save_nature_figure(fig, filename, formats=('pdf', 'png'), dpi=300,
                       ghost_text=False, align=True):
    """Save a figure in Nature-compliant formats.

    By default exports both PDF (vector, for Nature submission) and PNG
    (raster, for Word embedding / co-author sharing / preprints).

    Parameters
    ----------
    fig : matplotlib Figure
    filename : str
        Base filename without extension.
    formats : tuple of str
        Any of 'pdf', 'eps', 'tiff', 'png', 'svg'.
        Default is ('pdf', 'png').
    dpi : int
        Resolution for raster formats (default 300).
    ghost_text : bool
        If True, also save a "*_ghost" variant where all text is set to
        white (invisible on white background) while preserving identical
        layout/spacing. Useful for post-hoc editing in PowerPoint or
        Illustrator: overlay your own text knowing the bounding boxes
        match the labeled version exactly.
    """
    # Alignment groups labels by gridspec column. A figure whose root gridspec has one
    # column therefore counts every panel as the same column and drags all of its x
    # labels onto the lowest row. Callers that nest rows align their own.
    if align:
        fig.align_labels()

    # --- Standard export (labeled) ---
    for fmt in formats:
        path = f'{filename}.{fmt}'
        kwargs = {'format': fmt, 'dpi': dpi}
        if fmt == 'tiff':
            kwargs['pil_kwargs'] = {'compression': 'tiff_lzw'}
        fig.savefig(path, **kwargs)
        print(f'Saved: {path}')

    # --- Ghost-text export (invisible labels, same spacing) ---
    if ghost_text:
        _save_ghost_variant(fig, filename, formats, dpi)


def _collect_texts(fig):
    """Recursively collect all Text objects in a figure."""
    texts = []
    texts.extend(fig.texts)                        # suptitle, fig-level text
    for ax in fig.get_axes():
        texts.append(ax.title)                     # ax.set_title()
        texts.append(ax.xaxis.label)               # xlabel
        texts.append(ax.yaxis.label)               # ylabel
        texts.extend(ax.get_xticklabels())         # xtick labels
        texts.extend(ax.get_yticklabels())         # ytick labels
        texts.extend(ax.texts)                     # ax.text() annotations
        leg = ax.get_legend()
        if leg is not None:
            texts.extend(leg.get_texts())           # legend entries
            title = leg.get_title()
            if title:
                texts.append(title)
    return texts


def _save_ghost_variant(fig, filename, formats, dpi):
    """Save a version with all text set to white (invisible).

    The text objects remain in place with the same font and size so the
    layout (axis tick spacing, label bounding boxes, legend width) is
    identical. This makes it safe to overlay custom text in PowerPoint
    or Illustrator without the figure shifting.
    """
    texts = _collect_texts(fig)

    # Store originals
    originals = [(t, t.get_color()) for t in texts if t.get_text()]

    # Set all text to white
    for t, _ in originals:
        t.set_color('white')

    for fmt in formats:
        path = f'{filename}_ghost.{fmt}'
        kwargs = {'format': fmt, 'dpi': dpi}
        if fmt == 'tiff':
            kwargs['pil_kwargs'] = {'compression': 'tiff_lzw'}
        fig.savefig(path, **kwargs)
        print(f'Saved (ghost): {path}')

    # Restore originals so the figure object stays usable
    for t, original_color in originals:
        t.set_color(original_color)


# ---------------------------------------------------------------------------
# Seaborn integration
# ---------------------------------------------------------------------------
def set_seaborn_nature():
    """Configure seaborn to use Nature defaults.

    Call after apply_nature_style(). Sets seaborn context and palette
    to match Nature rcParams.
    """
    try:
        import seaborn as sns
    except ImportError:
        print('seaborn not installed; skipping seaborn configuration')
        return

    sns.set_context('paper', rc={
        'font.size': 7,
        'axes.labelsize': 7,
        'axes.titlesize': 7,
        'xtick.labelsize': 6,
        'ytick.labelsize': 6,
        'legend.fontsize': 6,
    })
    sns.set_palette(WONG)
    sns.set_style('ticks', rc={
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': False,
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    })
