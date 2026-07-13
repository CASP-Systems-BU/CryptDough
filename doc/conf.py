# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import subprocess

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'cdough'
copyright = '2026, BU CASP Lab'
author = 'BU CASP Lab'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [ "breathe", "myst_parser" ]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

breathe_projects = {"cdough": "xml"}
breathe_default_project = "cdough"

source_suffix = ['.rst', '.md']

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

html_theme_options = {
    "collapse_navigation": False,  # Expand all sections
    "navigation_depth": -1,         # How deep nested headings appear
    "titles_only": False           # Show nested headings as well
}

def get_git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).strip().decode("utf-8")
    except Exception:
        return "unknown"

commit = get_git_commit()

commit_url = f"https://github.com/CASP-Systems-BU/cdough/commit/{commit}"

rst_epilog = f"""
.. |release| replace:: ``{commit}``
"""