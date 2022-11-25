# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line.
SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
SOURCEDIR     = source
BUILDDIR      = _build
IMGDIR        = source/_static/imgs
BUILDIMGDIR   = _build/html/imgs
CODEIMGDIR   = _build/html/_static

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile


html:
	# cp README.md to docs, modify response-link
	cp -f "../README.md" "./source/README.md"
	cp -f "../SECURITY.md" "./source/SECURITY.md"
	cp -f "./source/README.md" "./source/README.md.tmp"
	sed 's/.md/.html/g; s/.\/docs\/source\//.\//g; s/.\/neural_coder\/extensions\/screenshots/imgs/g; s/.\/docs\/source\/_static/..\/\/_static/g; s/.\/examples/https:\/\/github.com\/intel\/neural-compressor\/tree\/master\/examples/g; s/.md/.html/g; ' "./source/README.md.tmp" > "./source/README.md"
	rm -f "./source/README.md.tmp"

	# modify docList
	cp -f "./source/doclist.rst" "./source/doclist.rst.tmp"
	sed 's/.md/.html/g;' "./source/doclist.rst.tmp" > "./source/doclist.rst"
	rm -f "./source/doclist.rst.tmp"

	
	# make sure other png can display normal
	$(SPHINXBUILD) -b html "$(SOURCEDIR)" "$(BUILDDIR)/html" $(SPHINXOPTS) $(O)

	cp source/_static/index.html $(BUILDDIR)/html/index.html
	mkdir -p "$(BUILDIMGDIR)"
	# common svg
	cp -f "$(CODEIMGDIR)/imgs/common/code.svg" "$(CODEIMGDIR)/images/view-page-source-icon.svg"	
	cp -f "$(CODEIMGDIR)/imgs/common/right.svg" "$(CODEIMGDIR)/images/chevron-right-orange.svg"	

	cp "../neural_coder/extensions/screenshots/extmanager.png"  "$(BUILDIMGDIR)/extmanager.png"
	cp "$(IMGDIR)/INC_GUI.gif" "$(BUILDIMGDIR)/INC_GUI.gif"
	cp "$(IMGDIR)/release_data.png" "$(BUILDIMGDIR)/release_data.png"


# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)