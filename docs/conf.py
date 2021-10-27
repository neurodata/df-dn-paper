# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))


# -- Project information -----------------------------------------------------

project = "DF/DN"
copyright = "2021, NeuroData"
author = "NeuroData"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = []

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------
# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]
html_static_path = ["_static"]
html_extra_path = []
modindex_common_prefix = ["dfdn."]
html_last_updated_fmt = "%b %d, %Y"
add_function_parentheses = False
html_show_sourcelink = False
html_show_sphinx = True
html_show_copyright = True

pygments_style = "sphinx"
smartquotes = False

# Use RTD Theme
import sphinx_rtd_theme

html_theme = "sphinx_rtd_theme"
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
html_theme_options = {
    "collapse_navigation": True,
    "navigation_depth": 4,
}

html_context = {
    "menu_links_name": "Useful Links",
    "menu_links": [
        (
            '<i class="fa fa-book fa-fw"></i> Preprint',
            "https://arxiv.org/abs/2108.13637",
        ),
        (
            '<i class="fa fa-external-link-square fa-fw"></i> Neurodata',
            "https://neurodata.io/",
        ),
        (
            '<i class="fa fa-gavel fa-fw"></i> Code of Conduct',
            "https://neurodata.io/about/codeofconduct/",
        ),
        (
            '<i class="fa fa-exclamation-circle fa-fw"></i> Issue Tracker',
            "https://github.com/neurodata/df-dn-paper/issues",
        ),
        (
            '<i class="fa fa-github fa-fw"></i> Source Code',
            "https://github.com/neurodata/df-dn-paper",
        ),
    ],
    # Enable the "Edit in GitHub link within the header of each page.
    "display_github": True,
    # Set the following variables to generate the resulting github URL for each page.
    # Format Template: https://{{ github_host|default("github.com") }}/{{ github_user }}
    # /{{ github_repo }}/blob/{{ github_version }}{{ conf_py_path }}{{ pagename }}
    # {{ suffix }}
    "github_user": "neurodata",
    "github_repo": "df-dn-paper",
    "github_version": "main/",
    "conf_py_path": "docs/",
}


html_css_files = [
    "custom.css",
]
