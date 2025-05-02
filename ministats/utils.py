from contextlib import contextmanager
import logging
import os

import matplotlib.pyplot as plt




@contextmanager
def loglevel(level, module=None):
    """
    Context manager to set logging level locally.
    Useful for silencing the output of Bambi model fit method.
    """
    if isinstance(level, str):
        LEVEL_NAMES_MAPPING = {
            'CRITICAL': 50, 'FATAL': 50, 'ERROR': 40, 'WARN': 30,
            'WARNING': 30, 'INFO': 20, 'DEBUG': 10, 'NOTSET': 0
        }
        level = level.upper()
        level = LEVEL_NAMES_MAPPING[level]
    logger = logging.getLogger(module)
    current_level = logger.getEffectiveLevel()
    logger.setLevel(level)
    try:
        yield
    finally:
        logger.setLevel(current_level)


def ensure_containing_dir_exists(filepath):
    parent = os.path.join(filepath, os.pardir)
    absparent = os.path.abspath(parent)
    if not os.path.exists(absparent):
        os.makedirs(absparent)


def default_labeler(params, params_to_latex):
    """
    Returns string appropriate for probability distribution label used in plot.
    """
    DEFAULT_PARAMS_TO_LATEX = {
        'mu': '\\mu',
        'sigma': '\\sigma',
        'lambda': '\\lambda',
        'beta': '\\beta',
        'a': 'a',
        'b': 'b',
        'N': 'N',
        'K': 'K',
        'k': 'k',
        'n': 'n',
        'p': 'p',
        'r': 'r',
    }
    params_to_latex = dict(DEFAULT_PARAMS_TO_LATEX, **params_to_latex)
    label_parts = []
    for param, value in params.items():
        if param in params_to_latex:
            label_part = '$' + params_to_latex[param] + '=' + str(value) + '$'
        else:
            label_part = str(param) + '=' + str(value)
        label_parts.append(label_part)
    label = ', '.join(label_parts)
    return label


def savefigure(obj, filename, tight_layout_kwargs=None):
    """
    Save the figure associated with `obj` (axes or figure).
    Assumes `filename` is relative path to pdf to save to,
    e.g. `figures/stats/some_figure.pdf`.
    """
    ensure_containing_dir_exists(filename)
    if not filename.endswith(".pdf"):
        filename = filename + ".pdf"

    if isinstance(obj, plt.Axes):
        fig = obj.figure
    elif isinstance(obj, plt.Figure):
        fig = obj
    else:
        raise ValueError("First argument must be Matplotlib figure or axes")

    # remove surrounding whitespace as much as possible
    if tight_layout_kwargs:
        fig.tight_layout(**tight_layout_kwargs)
    else:
        fig.tight_layout()

    # save as PDF
    fig.savefig(filename, dpi=300, bbox_inches="tight", pad_inches=0)
    print("Saved figure to", filename)

    # save as PNG
    filename2 = filename.replace(".pdf", ".png")
    fig.savefig(filename2, dpi=300, bbox_inches="tight", pad_inches=0)
    print("Saved figure to", filename2)
