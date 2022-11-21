# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line.
SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
SOURCEDIR     = source
BUILDDIR      = _build
IMGDIR        = source/_static/imgs
BUILDIMGDIR   = _build/html/io_doc/imgs
CODEIMGDIR   = _build/html/_static

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile


html:
	# cp README.md to docs, modify response-link
	cp -f "../README.md" "./source/io_doc/README.md"
	cp -f "./source/io_doc/README.md" "./source/io_doc/README.md.tmp"
	sed 's/.md/.html/g; s/.\/docs\/source\/io_doc/./g; s/.\/examples/..\/..\/..\/..\/examples/g; s/.\/neural_coder/..\/..\/..\/..\/neural_coder/g; s/.\/docs\/source\/_static/..\/\/_static/g;' "./source/io_doc/README.md.tmp" > "./source/io_doc/README.md"
	rm -f "./source/io_doc/README.md.tmp"
	
	# make sure other png can display normal
	$(SPHINXBUILD) -b html "$(SOURCEDIR)" "$(BUILDDIR)/html" $(SPHINXOPTS) $(O)

	cp source/_static/index.html $(BUILDDIR)/html/index.html
	mkdir -p "$(BUILDIMGDIR)"
	# common svg
	cp -f "$(CODEIMGDIR)/imgs/common/code.svg" "$(CODEIMGDIR)/images/view-page-source-icon.svg"	
	cp -f "$(CODEIMGDIR)/imgs/common/right.svg" "$(CODEIMGDIR)/images/chevron-right-orange.svg"	
	# README.md imgs
	# README.md link

	cp "../neural_coder/extensions/screenshots/extmanager.png"  "_build/html/_static/imgs/extmanager.png"
	cp "$(IMGDIR)/INC_GUI.gif" "$(BUILDIMGDIR)/INC_GUI.gif"	
	cp "$(IMGDIR)/release_data.png" "$(BUILDIMGDIR)/release_data.png"	
	# cp docs/imgs/architecture.png "$(BUILDDIR)/html/docs/imgs/architecture.png"
	# cp docs/imgs/workflow.png "$(BUILDDIR)/html/docs/imgs/workflow.png"	
	# cp docs/imgs/INC_GUI.gif "$(BUILDDIR)/html/docs/imgs/INC_GUI.gif"	
	# cp docs/imgs/release_data.png "$(BUILDDIR)/html/docs/imgs/release_data.png"	
	# cp "$(BUILDDIR)/html/README.html" "$(BUILDDIR)/html/README.html.tmp"
	# sed 's/.md/.html/g' "$(BUILDDIR)/html/README.html.tmp" > "$(BUILDDIR)/html/README.html"
	# rm -f "$(BUILDDIR)/html/README.html.tmp"


# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)