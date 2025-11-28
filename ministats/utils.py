from contextlib import contextmanager
import json
import logging
import matplotlib.pyplot as plt
import os
import shutil
import urllib.parse
import urllib.request



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




# GITHUB DOWNLOAD HELPERS
################################################################################

def github_api_request(path, params=None):
    """
    Make a GET request to the GitHub REST API at the given path.
    No authentication (sufficient for occasional use).
    """
    base_url = "https://api.github.com"
    url = base_url + path
    if params:
        url += "?" + urllib.parse.urlencode(params)
    headers = {
        "User-Agent": "python-stdlib-client",
        "Accept": "application/vnd.github.v3+json",
    }
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req) as resp:
        if resp.status != 200:
            raise RuntimeError(
                f"GitHub API error {resp.status}: {resp.read().decode(errors='replace')}"
            )
        return json.load(resp)


def list_dir_recursive(owner, repo, path, branch="main"):
    """
    Recursively list all files under `path` in the given repo+branch.
    Returns a list of file paths relative to the repo root.
    """
    api_path = f"/repos/{owner}/{repo}/contents/{path}"
    items = github_api_request(api_path, params={"ref": branch})
    files = []
    for item in items:
        item_type = item.get("type")
        item_path = item.get("path")
        if item_type == "file":
            files.append(item_path)
        elif item_type == "dir":
            files.extend(list_dir_recursive(owner, repo, item_path, branch=branch))
        # ignore symlinks, submodules etc.
    return files


def download_file_from_raw(owner, repo, branch, prefix, file_path, download_root):
    """
    Download file from GitHub and save to `download_root`.
    Remove the `prefix` from the repo path to avoid adding subdirectory.
    """
    # Compute relative local path
    prefix_slash = prefix + "/"
    if file_path.startswith(prefix_slash):
        relative = file_path[len(prefix_slash):]
    else:
        relative =  os.path.basename(file_path)
    local_path = os.path.join(download_root, relative)
    raw_url = (
        f"https://raw.githubusercontent.com/"
        f"{owner}/{repo}/{branch}/{file_path}"
    )
    # Ensure subdirectories exist
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    with urllib.request.urlopen(raw_url) as resp, open(local_path, "wb") as f:
        chunk = resp.read(8192)
        while chunk:
            f.write(chunk)
            chunk = resp.read(8192)
    return local_path


def download_files(owner, repo, branch, prefix, file_paths, download_root):
    """
    Download all files in `file_paths` to `download_root`,
    preserving their relative paths.
    """
    downloaded = []
    for fp in file_paths:
        local_path = download_file_from_raw(owner, repo, branch, prefix, fp, download_root)
        downloaded.append(local_path)
    return downloaded


def have_required_files(download_root, required_filenames):
    """
    Check if `download_root` exists and contains all `required_filenames`.
    """
    if not os.path.isdir(download_root):
        return False
    for name in required_filenames:
        path = os.path.join(download_root, name)
        if not os.path.isfile(path):
            return False
    return True




# DATASET AND SIMDATA DOWNLOAD HELPERS
################################################################################

OWNER = "minireference"
REPO = "noBSstats"
BRANCH = "main"


def ensure_datasets(verbose=False):
    """
    Make sure the `datasets/` folder with data files for
    the book is present in the current directory and.
    Otherwise:
    1. If ../datasets/ exists and contains the required files,
       copy from ../datasets/ to ./datasets/
    2. Download the files from GitHub repo:
       https://github.com/minireference/noBSstats/tree/main/datasets
    """
    DATASETS_PATH = "datasets"
    DATASETS_DOWNLOAD_ROOT = "datasets"
    PARENT_DATASETS = os.path.join("..", "datasets")
    REQUIRED_FILES = ["apples.csv", "doctors.csv", "eprices.csv"]    
    if have_required_files(DATASETS_DOWNLOAD_ROOT, REQUIRED_FILES):
        print(f"{DATASETS_DOWNLOAD_ROOT}/ directory already exists.")
    elif have_required_files(PARENT_DATASETS, REQUIRED_FILES):
        # Copy files from ../datasets/ to ./datasets/ (when working with .zip download)
        if verbose:
            print(f"Found valid datasets in {PARENT_DATASETS}/, copying recursively...")
        # Ensure local datasets folder exists
        os.makedirs(DATASETS_DOWNLOAD_ROOT, exist_ok=True)
        # Recursively copy contents of ../datasets/ into ./datasets/
        for root, dirs, files in os.walk(PARENT_DATASETS):
            # Compute relative path inside ../datasets/
            rel = os.path.relpath(root, PARENT_DATASETS)
            target_root = (DATASETS_DOWNLOAD_ROOT if rel == "." else os.path.join(DATASETS_DOWNLOAD_ROOT, rel) )
            os.makedirs(target_root, exist_ok=True)
            # Copy all files
            for fname in files:
                src = os.path.join(root, fname)
                dst = os.path.join(target_root, fname)
                shutil.copy2(src, dst)
                if verbose:
                    print(f"  copied {src} -> {dst}")
        print(f"Found {PARENT_DATASETS}/ and copied files to {DATASETS_DOWNLOAD_ROOT}/.")
    else:
        # Download files from GitHub
        all_files = list_dir_recursive(OWNER, REPO, DATASETS_PATH, branch=BRANCH)
        if verbose:
            print("Data files found in noBSstats repo:")
            for f in all_files:
                print("  ", f)
        downloaded_paths = download_files(
            OWNER, REPO, BRANCH, DATASETS_PATH, all_files, DATASETS_DOWNLOAD_ROOT
        )
        if verbose:
            print("\nDownloaded files:")
            for p in downloaded_paths:
                print("  ", p)
        print(f"Downloaded data files to {DATASETS_DOWNLOAD_ROOT}/ from GitHub.")


def ensure_simdata(force=False, verbose=False):
    """
    Download the simulation data used in the notebooks from:
    https://github.com/minireference/noBSstats/tree/main/notebooks/simdata
    Files are downloaded into the local directory:
        simdata/
    and flattened so that only the filenames (no subdirectories) are kept.
    """
    SIMDATA_PATH = "notebooks/simdata"  # directory within the repo
    SIMDATA_DOWNLOAD_ROOT = "simdata"        # local directory for simdata
    # Only hit the API if forced or the local directory doesn't exist yet
    if force or not os.path.isdir(SIMDATA_DOWNLOAD_ROOT):
        # List all files in notebooks/simdata
        all_files = list_dir_recursive(OWNER, REPO, SIMDATA_PATH, branch=BRANCH)
        if verbose:
            print("Simdata files found in noBSstats repo:")
            for f in all_files:
                print("  ", f)
        # Download files, flattening under SIMDATA_PATH
        downloaded_paths = download_files(
            OWNER, REPO, BRANCH, SIMDATA_PATH, all_files, SIMDATA_DOWNLOAD_ROOT
        )
        if verbose:
            print("\nDownloaded simdata files:")
            for p in downloaded_paths:
                print("  ", p)
    else:
        if verbose:
            print(f"{SIMDATA_DOWNLOAD_ROOT}/ already exists; skipping download.")
    print(f"{SIMDATA_DOWNLOAD_ROOT}/ directory present and ready.")

