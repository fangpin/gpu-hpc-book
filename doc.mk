# Documentation toolchain targets. Designed to be either included from a
# project's own Makefile (`include doc.mk`) or invoked directly:
#   make -f doc.mk docs DOC=<feishu-doc-url>

VENV := .venv
PY := $(VENV)/bin/python
SPHINX := $(VENV)/bin/sphinx-build

# sphinx 8.x needs python >= 3.11, which is newer than the `python3` shipped
# with macOS. Override with `make PYTHON=/path/to/python3.12 ...` if needed.
PYTHON ?= $(shell for p in python3.13 python3.12 python3.11 python3; do \
	command -v $$p >/dev/null 2>&1 || continue; \
	$$p -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)' 2>/dev/null || continue; \
	command -v $$p; break; \
done)

.PHONY: docs-install docs-sync docs-html docs docs-serve docs-clean

docs-install:
	@test -n "$(PYTHON)" || { echo "no python >= 3.11 found; install one (brew install python@3.12) or pass PYTHON=<path>"; exit 1; }
	$(PYTHON) -m venv $(VENV)
	$(VENV)/bin/pip install --upgrade pip
	$(VENV)/bin/pip install -r requirements-docs.txt
	@command -v lark-cli >/dev/null 2>&1 && echo "lark-cli: $(shell lark-cli --version 2>/dev/null | head -1)" || { \
		echo "lark-cli not found, installing @larksuite/cli via npm..."; \
		command -v npm >/dev/null 2>&1 || { echo "npm is required to install lark-cli"; exit 1; }; \
		npm install -g @larksuite/cli; \
	}
	@echo "hint: 如果尚未登录飞书，请运行 lark-cli auth login"

ifndef FROM
docs-sync:
	$(PY) doc_scripts/sync_lark_doc.py $(if $(DOC),--doc "$(DOC)")
else
docs-sync:
	$(PY) doc_scripts/sync_lark_doc.py --from-file "$(FROM)"
endif

docs-html:
	$(SPHINX) -b html docs/source docs/_build/html

docs-export: docs-html
	$(PY) doc_scripts/export_platform_posts.py --image-base "$(or $(IMAGE_BASE),pages)"

docs: docs-sync docs-html docs-export
	@echo "open docs/_build/html/index.html"

docs-serve: docs-html
	$(PY) -m http.server 8000 --directory docs/_build/html

docs-clean:
	rm -rf docs/_build
