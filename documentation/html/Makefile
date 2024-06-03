# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = source
BUILDDIR      = build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# manual 
github_docs:
    cd documentation/html
    rm -rf build
    mkdir ./build && touch ./build/.nojekyll
    @cp -a source/index.rst /source/index.rst
    @make -C ./build html
    @cp -a ./build/_build/html/. ./docs

# automatic github action push or pull request
github_action_docs:
    cd documentation/html
    rm -rf build
    mkdir build && touch build/.nojekyll
    @cp -a index.rst source/index.rst
    rm -rf build/_build && mkdir build/_build 
    rm -rf build/_autosummary
    pipx run poetry run sphinx-build -b html build build/_build/html
    @cp -a build/_build/html/* docs
