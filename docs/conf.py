from importlib import metadata

project = "Bergson"
copyright = "2025, EleutherAI"
html_title = "Bergson"
html_logo = "_static/bergson_logo.png"
html_favicon = "_static/favicon.ico"

author = (
    "Lucia Quirke"
    " Nora Belrose,"
    " Louis Jaburi,"
    " Baber Abbasi,"
    " William Li,"
    " Stella Biderman,"
)

extensions = [
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.autosectionlabel",
    "sphinx_autodoc_typehints",
    "sphinx.ext.doctest",
    "myst_parser",
    "nbsphinx",
]

release = metadata.version("Bergson")

napoleon_google_docstring = True
napoleon_use_param = False
napoleon_use_ivar = True

templates_path = ["_templates"]
exclude_patterns = ["build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints"]

html_theme = "furo"
html_static_path = ["_static"]
html_theme_options = {
    "source_repository": "https://github.com/EleutherAI/bergson",
    "source_branch": "main",
    "source_directory": "docs",
    "light_css_variables": {
        "sidebar-item-font-size": "85%",
    },
}
