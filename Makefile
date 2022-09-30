# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line.
SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
SPHINXPROJ    = ProjectnameIntelLowPrecisionOptimizationTool
SOURCEDIR     = .
BUILDDIR      = _build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile


html:
	$(SPHINXBUILD) -b html "$(SOURCEDIR)" "$(BUILDDIR)/html" $(SPHINXOPTS) $(O)
	cp _static/index.html $(BUILDDIR)/html/index.html
	mkdir "$(BUILDDIR)/html/docs/imgs"
	cp docs/imgs/architecture.png "$(BUILDDIR)/html/docs/imgs/architecture.png"
	cp docs/imgs/workflow.png "$(BUILDDIR)/html/docs/imgs/workflow.png"	
	cp docs/imgs/INC_GUI.gif "$(BUILDDIR)/html/docs/imgs/INC_GUI.gif"	
	cp docs/imgs/release_data.png "$(BUILDDIR)/html/docs/imgs/release_data.png"	
	cp "$(BUILDDIR)/html/README.html" "$(BUILDDIR)/html/README.html.tmp"
	sed 's/.md/.html/g' "$(BUILDDIR)/html/README.html.tmp" > "$(BUILDDIR)/html/README.html"
	rm -f "$(BUILDDIR)/html/README.html.tmp"


# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)