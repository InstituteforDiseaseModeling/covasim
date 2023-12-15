# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line.
ifeq ($(BUILD_DEBUG), 1)
	SPHINXOPTS = -v
else
	SPHINXOPTS = -jauto  # Use -v to debug instead
endif

SPHINXBUILD   = sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build
AUTODIR       = api/_autosummary

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

.PHONY: clean
clean: # The -f is to ignore missing files, not to force remove
	rm -rf $(BUILDDIR)
	rm -rf $(AUTODIR)
	cd tutorials; ./clean_outputs

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
